(function () {
  const host = new URLSearchParams(location.search).get("host") || "";

  const pageError = document.getElementById("page-error");
  const botLine = document.getElementById("bot-line");
  const gameList = document.getElementById("game-list");
  const gameRail = document.querySelector(".game-list-wrap");
  const railTitle = document.getElementById("rail-title");
  const reviewLog = document.getElementById("review-log");
  const swapButton = document.querySelector(".fold.swap");
  const gameEl = document.getElementById("game");
  const gameTitle = document.getElementById("game-title");
  const gameSub = document.getElementById("game-sub");
  const analyseButton = document.getElementById("analyse");
  const progressEl = document.getElementById("progress");
  const progressFill = document.getElementById("progress-fill");
  const analysisNote = document.getElementById("analysis-note");
  const reviewEval = document.getElementById("review-eval");
  const reviewSide = document.getElementById("review-side");
  const reviewLinesInput = document.getElementById("review-lines");
  const drawCount = document.getElementById("draw-count");
  const topCard = document.getElementById("top-card");
  const discardUnderEl = document.getElementById("discard-under");
  const stateLine = document.getElementById("state-line");
  const slider = document.getElementById("slider");
  const moveLine = document.getElementById("move-line");
  const moveAnalysis = document.getElementById("move-analysis");
  const evalName0 = document.getElementById("eval-name-0");
  const evalName1 = document.getElementById("eval-name-1");
  const evalBar = document.getElementById("eval-bar");
  const evalTop = document.getElementById("eval-top");
  const evalBottom = document.getElementById("eval-bottom");
  const evalPct = document.getElementById("eval-pct");
  const reviewCharts = document.getElementById("review-charts");
  const chartChance = document.getElementById("chart-chance");
  const chartSpread = document.getElementById("chart-spread");
  const spreadChart = document.getElementById("spread-chart");
  const legendP1 = document.getElementById("legend-p1");
  const legendP2 = document.getElementById("legend-p2");
  const legendSpreadP1 = document.getElementById("legend-spread-p1");
  const legendSpreadP2 = document.getElementById("legend-spread-p2");
  const legendEvalP1 = document.getElementById("legend-eval-p1");
  const legendEvalP2 = document.getElementById("legend-eval-p2");
  const chanceKeyP1 = document.getElementById("chance-key-p1");
  const chanceKeyP2 = document.getElementById("chance-key-p2");
  const chanceKeyEvalP1 = document.getElementById("chance-key-eval-p1");
  const chanceKeyEvalP2 = document.getElementById("chance-key-eval-p2");
  const spreadKeyP1 = document.getElementById("spread-key-p1");
  const spreadKeyP2 = document.getElementById("spread-key-p2");

  let current = null;
  let data = null;
  let analysis = null;
  let byIndex = {};
  let index = 0;
  let pollTimer = null;

  function esc(value) {
    return String(value).replace(/[&<>"']/g, function (ch) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[ch];
    });
  }

  function pct(value) {
    return value == null ? "–" : Math.round(value * 100) + "%";
  }

  // A difference in win chance, in percentage points.
  function points(value) {
    const text = Math.abs(value * 100).toFixed(1);
    return text + (text === "1.0" ? " point" : " points");
  }

  // The card an action plays (see prsi/actions.py), or null for "don't play".
  function actionCard(action) {
    if (action == null || action >= 44) return null;
    if (action < 28) {
      const ranks = [0, 1, 2, 3, 4, 6, 7];
      return Math.floor(action / 7) * 8 + ranks[action % 7];
    }
    return Math.floor((action - 28) / 4) * 8 + 5;
  }

  function api(path, options) {
    const url = path + (path.indexOf("?") >= 0 ? "&" : "?") + "host=" + encodeURIComponent(host);
    return fetch(url, options).then(function (response) {
      return response.json().then(function (body) {
        if (!response.ok) throw new Error(body.detail || response.statusText);
        return body;
      });
    });
  }

  function fail(error) {
    pageError.textContent = error.message || String(error);
    pageError.classList.remove("hidden");
  }

  function names() {
    return (data && data.record.names) || ["Player 1", "Player 2"];
  }

  function describeSeat(seat) {
    if (!seat) return "";
    if (seat.kind === "bot") return "bot" + (seat.network ? " · " + seat.network : " · " + seat.engine);
    return "human";
  }

  function lineLimit(input) {
    const count = Math.floor(Number(input && input.value));
    return count >= 1 ? count : 3;
  }

  function loadGames() {
    api("/api/review/games").then(function (body) {
      botLine.textContent = (body.bot_note || "") + (body.version ? " Analysis version: " + body.version + "." : "");
      gameList.replaceChildren();
      if (!body.games.length) {
        const empty = document.createElement("li");
        empty.className = "hint";
        empty.textContent = "No saved games yet.";
        gameList.appendChild(empty);
      }
      body.games.forEach(function (game) {
        const item = document.createElement("li");
        const button = document.createElement("button");
        button.type = "button";
        const who = (game.names || []).join(" vs ");
        let result = "draw";
        if (game.reason === "reset") result = "reset";
        else if (game.winner != null && game.names) {
          result = game.names[game.winner] + " won";
          if (game.reason === "resign") result += " · resigned";
        }
        const stamp = game.file.slice(0, 15).replace(/^(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2}).*/, "$1-$2-$3 $4:$5");
        button.innerHTML = esc(who) + "<small>" + esc(stamp) + " UTC · " + game.moves + " moves · " + esc(result) + "</small>";
        button.dataset.file = game.file;
        button.addEventListener("click", function () { openGame(game.file); });
        item.appendChild(button);
        gameList.appendChild(item);
      });
    }).catch(fail);
  }

  function openGame(file) {
    clearTimeout(pollTimer);
    current = file;
    analysis = null;
    byIndex = {};
    moveAnalysis.classList.add("hidden");
    reviewCharts.classList.add("hidden");
    analysisNote.textContent = "Loading…";
    Array.prototype.forEach.call(gameList.querySelectorAll("button"), function (button) {
      button.classList.toggle("active", button.dataset.file === file);
    });
    api("/api/review/game?file=" + encodeURIComponent(file)).then(function (body) {
      data = body;
      byIndex = {};
      const record = body.record;
      const seats = record.seats || [];
      gameTitle.textContent = names().join(" vs ");
      const result = record.result || {};
      let resultText = "Draw";
      if (result.reason === "reset") resultText = "Reset";
      else if (result.winner != null) {
        resultText = names()[result.winner] + " won";
        if (result.reason === "resign") resultText += " · resigned";
      }
      gameSub.textContent = names()[0] + " (" + describeSeat(seats[0]) + ") vs " + names()[1] +
        " (" + describeSeat(seats[1]) + ") · " + record.actions.length + " moves · " + resultText +
        (result.reason === "stuck" ? " (nobody could move)" : "");
      slider.max = String(body.frames.length - 1);
      gameEl.classList.remove("hidden");
      reviewEval.classList.remove("hidden");
      reviewSide.classList.remove("hidden");
      setAnalysis(body.analysis);
      show(0);
    }).catch(fail);
  }

  function setAnalysis(status) {
    clearTimeout(pollTimer);
    const state = status.status || "missing";
    progressEl.classList.add("hidden");
    if (state === "done") {
      const result = status.result;
      analysis = result;
      byIndex = {};
      result.moves.forEach(function (move) { byIndex[move.index] = move; });
      if (result.eval) {
        analyseButton.classList.add("hidden");
        analysisNote.textContent = "Analysed by " + result.version + " in " + result.seconds + " s.";
      } else {
        analyseButton.textContent = "Add the evaluation bar";
        analyseButton.classList.remove("hidden");
        analysisNote.textContent = "Analysed by " + result.version + " in " + result.seconds + " s. The evaluation bar has not been added yet.";
      }
    } else if (state === "running") {
      analyseButton.classList.add("hidden");
      progressEl.classList.remove("hidden");
      const fallback = data ? (status.phase === "bar" ? data.frames.length : data.record.actions.length) : 0;
      const total = status.total || fallback;
      const done = status.done || 0;
      progressFill.style.width = (total ? Math.round(100 * done / total) : 0) + "%";
      analysisNote.textContent = status.phase === "bar"
        ? "Scoring the bar… position " + done + " of " + (total || "?")
        : "Analysing… move " + done + " of " + (total || "?");
      const file = current;
      pollTimer = setTimeout(function () {
        api("/api/review/analysis?file=" + encodeURIComponent(file)).then(function (next) {
          if (file === current) setAnalysis(next);
        }).catch(fail);
      }, 1000);
    } else if (state === "unavailable") {
      analyseButton.classList.add("hidden");
      analysisNote.textContent = "The bot is not available on this server.";
    } else {
      const partial = status.result;
      const barOnly = partial && partial.moves && !partial.eval;
      analyseButton.textContent = barOnly ? "Add the evaluation bar" : "Analyze";
      analyseButton.classList.remove("hidden");
      analysisNote.textContent = state.indexOf("failed") === 0 ? "The analysis failed: " + state : "Not analysed yet";
      if (barOnly) {
        analysis = partial;
        byIndex = {};
        partial.moves.forEach(function (move) { byIndex[move.index] = move; });
      }
    }
    if (data) show(index);
  }

  function renderHand(seat, cards, playedCard, toMove) {
    const handEl = document.getElementById("hand-" + seat);
    const nameEl = document.getElementById("name-" + seat);
    nameEl.textContent = names()[seat] + " · " + cards.length + (cards.length === 1 ? " card" : " cards") + (toMove ? " · to move" : "");
    nameEl.classList.toggle("to-move", toMove);
    const left = handEl.scrollLeft;
    handEl.replaceChildren();
    cards.forEach(function (card) {
      const face = prsiCards.element(card);
      if (card.id === playedCard) face.classList.add("played");
      handEl.appendChild(face);
    });
    handEl.scrollLeft = left;
  }

  function show(i) {
    if (!data) return;
    index = Math.max(0, Math.min(i, data.frames.length - 1));
    slider.value = String(index);
    const frame = data.frames[index];
    const played = actionCard(frame.action);
    renderHand(0, frame.hands[0], frame.current === 0 ? played : null, frame.current === 0);
    renderHand(1, frame.hands[1], frame.current === 1 ? played : null, frame.current === 1);
    discardUnderEl.replaceChildren();
    (frame.under_cards || []).forEach(function (card, place) {
      const under = prsiCards.element(card);
      under.classList.add("under-" + place);
      discardUnderEl.appendChild(under);
    });
    prsiCards.paint(topCard, frame.top_card, {
      ribbon: frame.named_by_over ? frame.active_suit_name : "",
      ribbonSuit: frame.active_suit
    });
    drawCount.textContent = String(frame.draw_pile_size);
    let state = "";
    if (frame.penalty_cards) state = "A 7 is active (" + frame.penalty_cards + " cards).";
    else if (frame.ace_active) state = "An Ace is active.";
    stateLine.textContent = state;
    stateLine.classList.toggle("hidden", !state);

    const total = data.frames.length - 1;
    const move = byIndex[index];
    if (frame.action != null) {
      moveLine.textContent = "Move " + (index + 1) + " of " + total + " · " + names()[frame.current] + ": " + frame.label;
    } else {
      const result = frame.result || (data.record && data.record.result) || {};
      if (result.reason === "reset") moveLine.textContent = "Hand stopped. The score was reset.";
      else if (result.reason === "resign") {
        moveLine.textContent = (names()[result.winner] || "Someone") + " wins · resigned";
      } else {
        moveLine.textContent = "End of the game · " + (result.winner == null ? "draw" : names()[result.winner] + " wins");
      }
    }
    renderMoveAnalysis(frame, move);
    renderEval();
    renderCharts();
    renderReviewLog();
    window.prsiCards.fit(document.querySelector(".review"));
  }

  function renderReviewLog() {
    const fromBottom = reviewLog.scrollHeight - reviewLog.scrollTop - reviewLog.clientHeight;
    const stick = fromBottom < 40;
    const kept = reviewLog.scrollTop;
    reviewLog.replaceChildren();
    if (!data || !data.log) return;
    const end = data.log_at ? data.log_at[index] : 0;
    const events = data.log.slice(0, end == null ? 0 : end);
    events.forEach(function (event) {
      const item = document.createElement("li");
      if (!prsiCards.formatEvent(item, event, null, names(), null)) return;
      const suit = prsiCards.lineSuit(event);
      if (suit) item.className = "suit-" + suit;
      reviewLog.appendChild(item);
    });
    reviewLog.scrollTop = stick ? reviewLog.scrollHeight : kept;
  }

  swapButton.addEventListener("click", function () {
    const moves = gameRail.classList.toggle("is-moves");
    railTitle.textContent = moves ? "Recent moves" : "Saved games";
    gameRail.setAttribute("aria-label", moves ? "Recent moves" : "Saved games");
    swapButton.setAttribute("aria-label", moves ? "Show saved games" : "Show recent moves");
    if (moves) reviewLog.scrollTop = reviewLog.scrollHeight;
  });

  // A scored bar glides. Empty, and the first score after empty, snap.
  function placeFill(topGrow, bottomGrow, glide) {
    if (!glide) evalBar.classList.add("no-glide");
    evalTop.style.flexGrow = String(topGrow);
    evalBottom.style.flexGrow = String(bottomGrow);
    if (!glide) {
      void evalBar.offsetWidth;
      evalBar.classList.remove("no-glide");
    }
  }

  function renderEval() {
    const who = names();
    evalName0.textContent = who[0] || "Player 1";
    evalName1.textContent = who[1] || "Player 2";
    const values = analysis && analysis.eval;
    const value = values && index < values.length ? values[index] : null;
    if (value == null) {
      placeFill(1, 1, false);
      evalBar.classList.add("empty");
      evalPct.textContent = "–";
      evalPct.classList.add("is-empty");
      evalBar.setAttribute("aria-label", "Win chance not scored yet");
      return;
    }
    const bottom = Math.max(0, Math.min(1, Number(value)));
    const glide = !evalBar.classList.contains("empty");
    placeFill(1 - bottom, bottom, glide);
    evalBar.classList.remove("empty");
    evalPct.textContent = pct(bottom);
    evalPct.classList.remove("is-empty");
    evalBar.setAttribute("aria-label", who[0] + " win chance " + pct(bottom));
  }

  function renderMoveAnalysis(frame, analysis) {
    if (frame.action == null) {
      moveAnalysis.classList.add("hidden");
      return;
    }
    moveAnalysis.classList.remove("hidden");
    const who = names()[frame.current];
    if (!analysis) {
      if (!Object.keys(byIndex).length) {
        moveAnalysis.classList.add("hidden");
        moveAnalysis.innerHTML = "";
        return;
      }
      moveAnalysis.classList.remove("hidden");
      moveAnalysis.innerHTML = "<p class=\"hint\">No analysis for this move.</p>";
      return;
    }
    if (analysis.forced && analysis.win_chance == null) {
      moveAnalysis.innerHTML = "<p class=\"hint\">" + esc(who) + " had only one legal move.</p>";
      return;
    }
    let html = "<div class=\"panel-head\"><h2>Bot's view for " + esc(who) + "</h2><p><span class=\"win\">" +
      pct(analysis.win_chance) + "</span> win chance · bot's choice: <b>" + esc(analysis.bot_label) + "</b> · " +
      Number(analysis.iterations).toLocaleString() + " imagined games</p></div>";
    html += "<table class=\"moves\"><thead><tr><th>Move</th><th>Win chance</th><th>Tried</th><th>Network</th></tr></thead><tbody>";
    analysis.moves.slice(0, lineLimit(reviewLinesInput)).forEach(function (move) {
      const classes = [];
      if (move.action === analysis.bot_action) classes.push("chosen");
      if (move.action === analysis.actual) classes.push("actual");
      html += "<tr class=\"" + classes.join(" ") + "\"><td>" + esc(move.label) + "</td><td>" + pct(move.win_chance) +
        "</td><td>" + Number(move.visits || 0).toLocaleString() + "</td><td>" + pct(move.prior) + "</td></tr>";
    });
    html += "</tbody></table>";
    if (analysis.forced) {
      html += "<p class=\"hint\">" + esc(who) + " had only one legal move.</p>";
    }
    if (analysis.loss != null && analysis.actual !== analysis.bot_action) {
      html += "<p class=\"hint\">The move played rates " + points(analysis.loss) + " below the bot's choice.</p>";
    }
    moveAnalysis.innerHTML = html;
  }

  function finite(value) {
    const number = Number(value);
    return Number.isFinite(number) ? number : null;
  }

  function anyValue(series) {
    return series.some(function (value) { return value != null; });
  }

  // Own-view win chance, held from each searched turn until the next one.
  function heldChances(frames) {
    const p1 = new Array(frames).fill(null);
    const p2 = new Array(frames).fill(null);
    let last1 = null;
    let last2 = null;
    const moves = (analysis && analysis.moves) || [];
    const at = {};
    moves.forEach(function (move) { at[move.index] = move; });
    for (let i = 0; i < frames; i++) {
      const move = at[i];
      const chance = move ? finite(move.win_chance) : null;
      if (chance != null && move.player === 0) last1 = chance;
      else if (chance != null && move.player === 1) last2 = chance;
      p1[i] = last1;
      p2[i] = last2;
    }
    return { p1: p1, p2: p2 };
  }

  function evalSeries(frames) {
    const values = new Array(frames).fill(null);
    const raw = analysis && analysis.eval;
    if (!raw) return values;
    for (let i = 0; i < frames && i < raw.length; i++) values[i] = finite(raw[i]);
    return values;
  }

  // The stored evaluation is the bottom player's chance. The top player's is its mirror.
  function mirrorEval(values) {
    return values.map(function (value) {
      return value == null ? null : 1 - value;
    });
  }

  // Player 2's bar is one minus the evaluation, so both spreads share a zero line.
  function spreadSeries(chances, evaluation) {
    const s1 = [];
    const s2 = [];
    for (let i = 0; i < evaluation.length; i++) {
      const ev = evaluation[i];
      s1.push(chances.p1[i] == null || ev == null ? null : chances.p1[i] - ev);
      s2.push(chances.p2[i] == null || ev == null ? null : chances.p2[i] - (1 - ev));
    }
    return { s1: s1, s2: s2 };
  }

  function spreadPeak(s1, s2) {
    let peak = 0;
    s1.concat(s2).forEach(function (value) {
      if (value != null) peak = Math.max(peak, Math.abs(value * 100));
    });
    const padded = Math.max(10, peak * 1.12);
    return Math.ceil(padded / 2) * 2;
  }

  function formatPoints(points) {
    const rounded = Math.round(points);
    if (rounded === 0) return "0";
    return (rounded > 0 ? "+" : "") + rounded;
  }

  function axisTicks(count) {
    if (count <= 1) return [0];
    if (count <= 6) {
      const all = [];
      for (let i = 0; i < count; i++) all.push(i);
      return all;
    }
    const ticks = [];
    for (let step = 0; step <= 4; step++) ticks.push(Math.round(step * (count - 1) / 4));
    return ticks.filter(function (value, i) { return ticks.indexOf(value) === i; });
  }

  function chartMarkup(spec) {
    const width = 640;
    const height = 180;
    const left = 48;
    const right = 12;
    const top = 16;
    const bottom = 26;
    const plotW = width - left - right;
    const plotH = height - top - bottom;
    const count = spec.count;
    function xOf(i) {
      if (count <= 1) return left + plotW / 2;
      return left + (i / (count - 1)) * plotW;
    }
    function yOf(value) {
      const span = spec.yMax - spec.yMin;
      const t = span === 0 ? 0.5 : (spec.yMax - value) / span;
      return top + t * plotH;
    }
    let body = "";
    spec.yTicks.forEach(function (tick) {
      const y = yOf(tick.value);
      const grid = tick.zero ? "grid zero" : "grid";
      body += "<line class=\"" + grid + "\" x1=\"" + left + "\" y1=\"" + y.toFixed(1) +
        "\" x2=\"" + (left + plotW) + "\" y2=\"" + y.toFixed(1) + "\"></line>";
      body += "<text x=\"" + (left - 8) + "\" y=\"" + (y + 4).toFixed(1) + "\" text-anchor=\"end\">" +
        esc(tick.label) + "</text>";
    });
    axisTicks(count).forEach(function (i) {
      const x = xOf(i);
      const anchor = i === 0 ? "start" : i === count - 1 ? "end" : "middle";
      body += "<text x=\"" + x.toFixed(1) + "\" y=\"" + (height - 8) + "\" text-anchor=\"" + anchor + "\">" +
        (i + 1) + "</text>";
    });
    spec.series.forEach(function (series) {
      const runs = [];
      let run = [];
      series.values.forEach(function (value, i) {
        if (value == null) {
          if (run.length) runs.push(run);
          run = [];
          return;
        }
        run.push(i);
      });
      if (run.length) runs.push(run);
      runs.forEach(function (points) {
        if (points.length === 1) {
          const i = points[0];
          body += "<circle class=\"" + series.cls + "\" cx=\"" + xOf(i).toFixed(1) +
            "\" cy=\"" + yOf(series.values[i]).toFixed(1) + "\" r=\"2.4\"></circle>";
          return;
        }
        const drawn = points.map(function (i) {
          return xOf(i).toFixed(1) + "," + yOf(series.values[i]).toFixed(1);
        }).join(" ");
        body += "<polyline class=\"" + series.cls + "\" points=\"" + drawn + "\"></polyline>";
      });
    });
    if (spec.marker != null && spec.marker >= 0 && spec.marker < count) {
      const x = xOf(spec.marker);
      body += "<line class=\"marker\" x1=\"" + x.toFixed(1) + "\" y1=\"" + top +
        "\" x2=\"" + x.toFixed(1) + "\" y2=\"" + (top + plotH) + "\"></line>";
    }
    return body;
  }

  function renderCharts() {
    if (!data || !analysis) {
      reviewCharts.classList.add("hidden");
      return;
    }
    const frames = data.frames.length;
    const chances = heldChances(frames);
    const evaluation = evalSeries(frames);
    const hasP1 = anyValue(chances.p1);
    const hasP2 = anyValue(chances.p2);
    const hasEval = anyValue(evaluation);
    if (!hasP1 && !hasP2 && !hasEval) {
      reviewCharts.classList.add("hidden");
      return;
    }
    const who = names();
    legendP1.textContent = who[0];
    legendP2.textContent = who[1];
    legendEvalP1.textContent = who[0] + ", both hands";
    legendEvalP2.textContent = who[1] + ", both hands";
    legendSpreadP1.textContent = who[0];
    legendSpreadP2.textContent = who[1];
    chanceKeyP1.classList.toggle("hidden", !hasP1);
    chanceKeyP2.classList.toggle("hidden", !hasP2);
    chanceKeyEvalP1.classList.toggle("hidden", !hasEval);
    chanceKeyEvalP2.classList.toggle("hidden", !hasEval);
    const chanceSeries = [];
    if (hasEval) {
      chanceSeries.push({ values: mirrorEval(evaluation), cls: "line-eval line-p2" });
      chanceSeries.push({ values: evaluation, cls: "line-eval line-p1" });
    }
    if (hasP2) chanceSeries.push({ values: chances.p2, cls: "line-p2" });
    if (hasP1) chanceSeries.push({ values: chances.p1, cls: "line-p1" });
    const chanceTicks = [0, 0.25, 0.5, 0.75, 1].map(function (value) {
      return { value: value, label: String(Math.round(value * 100)), zero: value === 0 };
    });
    chartChance.innerHTML = chartMarkup({
      count: frames,
      yMin: 0,
      yMax: 1,
      yTicks: chanceTicks,
      series: chanceSeries,
      marker: index
    });
    chartChance.setAttribute("aria-label",
      "Win chance across the game. Marker at position " + (index + 1) + " of " + frames + ".");
    const spread = spreadSeries(chances, evaluation);
    const hasSpread = hasEval && (anyValue(spread.s1) || anyValue(spread.s2));
    spreadChart.classList.toggle("hidden", !hasSpread);
    if (hasSpread) {
      const peak = spreadPeak(spread.s1, spread.s2);
      const spreadSeriesList = [];
      if (anyValue(spread.s2)) spreadSeriesList.push({ values: spread.s2, cls: "line-p2" });
      if (anyValue(spread.s1)) spreadSeriesList.push({ values: spread.s1, cls: "line-p1" });
      spreadKeyP1.classList.toggle("hidden", !anyValue(spread.s1));
      spreadKeyP2.classList.toggle("hidden", !anyValue(spread.s2));
      chartSpread.innerHTML = chartMarkup({
        count: frames,
        yMin: -peak / 100,
        yMax: peak / 100,
        yTicks: [peak, peak / 2, 0, -peak / 2, -peak].map(function (points) {
          return { value: points / 100, label: formatPoints(points), zero: points === 0 };
        }),
        series: spreadSeriesList,
        marker: index
      });
      chartSpread.setAttribute("aria-label",
        "Spread against the evaluation, in percentage points. Marker at position " + (index + 1) + " of " + frames + ".");
    }
    reviewCharts.classList.remove("hidden");
  }

  analyseButton.addEventListener("click", function () {
    if (!current) return;
    analyseButton.classList.add("hidden");
    api("/api/review/analyse?file=" + encodeURIComponent(current), { method: "POST" }).then(setAnalysis).catch(fail);
  });
  reviewLinesInput.addEventListener("input", function () { if (data) show(index); });
  slider.addEventListener("input", function () { show(Number(slider.value)); });
  document.getElementById("first").addEventListener("click", function () { show(0); });
  document.getElementById("prev").addEventListener("click", function () { show(index - 1); });
  document.getElementById("next").addEventListener("click", function () { show(index + 1); });
  document.getElementById("last").addEventListener("click", function () { show(Infinity); });
  document.addEventListener("keydown", function (event) {
    if (event.target && event.target.tagName === "INPUT" && event.target.type !== "range") return;
    if (event.key === "ArrowLeft") show(index - 1);
    if (event.key === "ArrowRight") show(index + 1);
  });

  if (!host) {
    fail(new Error("Open this page from the host link (it needs ?host=…)."));
  } else {
    loadGames();
  }
  window.prsiCards.watch(document.querySelector(".review"));
})();
