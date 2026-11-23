"""FastAPI app: one room, a websocket, and the static page."""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from json import JSONDecodeError
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.websockets import WebSocketDisconnect

from server.review import (
    analyse_game,
    fill_eval,
    game_frames,
    list_games,
    read_status,
    safe_game_path,
    spec_version,
)
from server.room import Room

log = logging.getLogger("prsi.server")
ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = ROOT / "static"
CARDS_DIR = STATIC_DIR / "cards"


class NoCacheStatic(StaticFiles):
    async def get_response(self, path, scope):
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-cache"
        return response


def create_app(room: Room, host: str = "127.0.0.1", port: int = 8000) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        _print_banner(app.state.room, app.state.host, app.state.port)
        yield
        if app.state.room.bot_factory is not None:
            from server.bot import shutdown

            shutdown()

    app = FastAPI(lifespan=lifespan)
    app.state.room = room
    app.state.host = host
    app.state.port = port

    @app.websocket("/ws")
    async def socket(websocket: WebSocket):
        await websocket.accept()
        player = None
        try:
            while True:
                try:
                    data = await websocket.receive_json()
                except WebSocketDisconnect:
                    break
                except (JSONDecodeError, ValueError):
                    await _error(websocket, "I didn't understand that.")
                    continue
                if not isinstance(data, dict):
                    await _error(websocket, "I didn't understand that.")
                    continue
                kind = data.get("type")
                if kind == "join":
                    if player is not None:
                        await _error(websocket, "You are already in the room.")
                        continue
                    if not room.matches(data.get("code", "")):
                        await _error(websocket, "Wrong room code.")
                        continue
                    player, err = await room.join(data.get("name", ""), websocket, data.get("host"))
                    if err or player is None:
                        player = None
                        await _error(websocket, err or "Could not join.")
                        continue
                    await websocket.send_json(
                        {
                            "type": "joined",
                            "token": player.token,
                            "seat": player.seat,
                            "name": player.name,
                            "host": player.is_host,
                        }
                    )
                    await room.publish()
                elif kind == "rejoin":
                    if player is not None:
                        await _error(websocket, "You are already in the room.")
                        continue
                    player, err = await room.rejoin(data.get("token", ""), websocket)
                    if err or player is None:
                        player = None
                        await _error(websocket, err or "Could not rejoin.")
                        continue
                    await websocket.send_json(
                        {
                            "type": "joined",
                            "token": player.token,
                            "seat": player.seat,
                            "name": player.name,
                            "host": player.is_host,
                        }
                    )
                    await room.publish()
                elif kind == "set_mode":
                    if player is None:
                        await _error(websocket, "Join the room first.")
                        continue
                    err = await room.set_mode(player, data.get("mode"))
                    if err:
                        await _error(websocket, err)
                elif kind == "action":
                    if player is None:
                        await _error(websocket, "Join the room first.")
                        continue
                    action = data.get("action")
                    move = data.get("move")
                    if type(action) is not int or type(move) is not int:
                        await _error(websocket, "Unknown move.")
                        continue
                    err = await room.submit_action(player.seat, action, move)
                    if err:
                        await _error(websocket, err)
                elif kind == "confirm_move":
                    if player is None:
                        await _error(websocket, "Join the room first.")
                        continue
                    err = await room.confirm_move(player, data.get("move"))
                    if err:
                        await _error(websocket, err)
                elif kind == "rematch":
                    if player is None:
                        await _error(websocket, "Join the room first.")
                        continue
                    if player.is_host and room.mode == 4:
                        await room.next_game(player)
                    else:
                        await room.vote_rematch(player.seat)
                elif kind == "reset_score":
                    if player is None:
                        await _error(websocket, "Join the room first.")
                        continue
                    err = await room.reset_score(player)
                    if err:
                        await _error(websocket, err)
                elif kind == "resign":
                    if player is None:
                        await _error(websocket, "Join the room first.")
                        continue
                    err = await room.resign(player)
                    if err:
                        await _error(websocket, err)
                else:
                    await _error(websocket, "Unknown message.")
        except WebSocketDisconnect:
            pass
        except Exception:
            log.exception("A connection failed.")
        finally:
            if player is not None:
                player.sockets.discard(websocket)
                await room.publish()

    app.state.analyses = {}

    def require_host(host: str) -> None:
        if not room.is_host_token(host):
            raise HTTPException(status_code=403, detail="This page needs the host link.")

    def game_path(name: str) -> Path:
        path = safe_game_path(room.games_dir, name)
        if path is None:
            raise HTTPException(status_code=404, detail="No such game.")
        return path

    @app.get("/review")
    async def review_page():
        return FileResponse(STATIC_DIR / "review.html", headers={"Cache-Control": "no-cache"})

    @app.get("/api/review/games")
    async def review_games(host: str = ""):
        require_host(host)
        spec = room.bot_spec
        return {
            "games": list_games(room.games_dir),
            "version": spec_version(spec) if spec else None,
            "bot_note": room.bot_note,
        }

    @app.get("/api/review/game")
    async def review_game(file: str, host: str = ""):
        require_host(host)
        data = game_frames(game_path(file))
        data["analysis"] = _analysis_status(app, room, file)
        return data

    @app.post("/api/review/analyse")
    async def review_analyse(file: str, host: str = ""):
        require_host(host)
        game_path(file)
        spec = room.bot_spec
        if spec is None:
            raise HTTPException(status_code=400, detail="The bot is not available.")
        status = _analysis_status(app, room, file)
        if status["status"] == "running":
            return status
        result = status.get("result") or {}
        if status["status"] == "done" and "eval" in result:
            return status
        from server.bot import executor

        key = (file, spec_version(spec))
        future = app.state.analyses.get(key)
        if future is not None and not future.done():
            return {"status": "running", "done": 0, "total": None, "phase": "bar" if "moves" in result else "moves"}
        # A saved move list only needs the bar. Anything else is a full analysis.
        job = fill_eval if "moves" in result else analyse_game
        phase = "bar" if job is fill_eval else "moves"
        loop = asyncio.get_running_loop()
        app.state.analyses[key] = loop.run_in_executor(
            executor(), job, str(room.games_dir), file, spec
        )
        return {"status": "running", "done": 0, "total": None, "phase": phase}

    @app.get("/api/review/analysis")
    async def review_analysis(file: str, host: str = ""):
        require_host(host)
        game_path(file)
        return _analysis_status(app, room, file)

    @app.get("/table-texture.png")
    async def table_texture():
        path = ROOT / "design elements" / "table-texture.png"
        if not path.is_file():
            raise HTTPException(status_code=404)
        return FileResponse(path, media_type="image/png")

    app.mount("/cards", StaticFiles(directory=CARDS_DIR), name="cards")
    app.mount("/", NoCacheStatic(directory=STATIC_DIR, html=True), name="static")
    return app


def _analysis_status(app: FastAPI, room: Room, file: str) -> dict:
    spec = room.bot_spec
    if spec is None:
        return {"status": "unavailable"}
    version = spec_version(spec)
    status = read_status(room.games_dir, file, version)
    future = app.state.analyses.get((file, version))
    result = status.get("result") or {}
    if future is not None and not future.done():
        if status.get("status") == "running":
            return status
        phase = "bar" if "moves" in result else "moves"
        return {"status": "running", "done": 0, "total": None, "phase": phase}
    if future is not None and future.done() and future.exception() is not None and status.get("status") != "done":
        failed = {"status": f"failed: {future.exception()}"}
        if status.get("result") is not None:
            failed["result"] = status["result"]
        return failed
    return status


async def _error(websocket: WebSocket, message: str) -> None:
    try:
        await websocket.send_json({"type": "error", "message": message})
    except Exception:
        pass


def _print_banner(room: Room, host: str, port: int) -> None:
    shown_host = "127.0.0.1" if host in ("0.0.0.0", "::") else host
    lines = [
        "Prší is running.",
        f"  On this computer:  http://{shown_host}:{port}",
        f"  Room code:         {room.code}",
        f"  Host link:         http://{shown_host}:{port}/?host={room.host_token}",
        "                     (private: only this link can choose the game mode,",
        "                      watch the bot and open the review page)",
    ]
    if room.bot_note:
        lines.append(f"  Bot:               {room.bot_note}")
    lines += [
        "",
        "Share the ngrok link and this room code with your friend.",
        "Never share the host link.",
        "Leave this window open while you play.",
    ]
    text = "\n".join(lines)
    try:
        print("\n" + text + "\n", flush=True)
    except UnicodeEncodeError:
        print("\n" + text.replace("Prší", "Prsi") + "\n", flush=True)
