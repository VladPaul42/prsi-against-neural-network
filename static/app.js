(function () {
  const params = new URLSearchParams(location.search);
  const hostToken = params.get("host") || "";
  const slot = hostToken ? "host" : (params.get("p") || "1");
  const storageKey = "prsi.token." + slot;
  const nameKey = "prsi.name." + slot;

  const joinScreen = document.getElementById("join");
  const joinForm = document.getElementById("join-form");
  const joinError = document.getElementById("join-error");
  const nameInput = document.getElementById("name");
  const codeInput = document.getElementById("code");
  const table = document.getElementById("table");
  const waiting = document.getElementById("waiting");
  const waitingText = document.getElementById("waiting-text");
  const waitingInvite = document.getElementById("waiting-invite");
  const board = document.getElementById("board");
  const waitingCode = document.getElementById("waiting-code");
  const roomLabel = document.getElementById("room-label");
  const scoreEl = document.getElementById("score");
  const linkStatus = document.getElementById("link-status");
  const oppName = document.getElementById("opp-name");
  const oppMeta = document.getElementById("opp-meta");
  const oppCards = document.getElementById("opp-cards");
  const youName = document.getElementById("you-name");
  const youMeta = document.getElementById("you-meta");
  const handEl = document.getElementById("hand");
  const discardEl = document.getElementById("discard");
  const discardUnderEl = document.getElementById("discard-under");
  const drawPileEl = document.getElementById("draw-pile");
  const drawCount = document.getElementById("draw-count");
  const turnLine = document.getElementById("turn-line");
  const passButton = document.getElementById("pass-button");
  const logEl = document.getElementById("log");
  const resultEl = document.getElementById("result");
  const resultTitle = document.getElementById("result-title");
  const resultDetail = document.getElementById("result-detail");
  const resultScore = document.getElementById("result-score");
  const rematchButton = document.getElementById("rematch");
  const rematchNote = document.getElementById("rematch-note");
  const picker = document.getElementById("picker");
  const pickerCard = document.getElementById("picker-card");
  const pickerSuits = document.getElementById("picker-suits");
  const help = document.getElementById("help");
  const toast = document.getElementById("toast");
  const hostBar = document.getElementById("host-bar");
  const modeButtons = document.getElementById("mode-buttons");
  const botNote = document.getElementById("bot-note");
  const reviewLink = document.getElementById("review-link");
  const resetScoreButton = document.getElementById("reset-score");
  const resignButton = document.getElementById("resign-button");
  const panelsEl = document.getElementById("panels");
  const panelsFold = document.getElementById("panels-fold");
  const playStage = document.getElementById("play-stage");
  const topLinesInput = document.getElementById("top-lines");
  const liveEval = document.getElementById("live-eval");
  const liveEvalName0 = document.getElementById("live-eval-name-0");
  const liveEvalName1 = document.getElementById("live-eval-name-1");
  const liveEvalBar = document.getElementById("live-eval-bar");
  const liveEvalTop = document.getElementById("live-eval-top");
  const liveEvalBottom = document.getElementById("live-eval-bottom");
  const liveEvalPct = document.getElementById("live-eval-pct");
  const liveEvalCount = document.getElementById("live-eval-count");

  let ws = null;
  let retryMs = 1000;
  let retryTimer = null;
  let lastSeq = 0;
  let moveId = 0;
  let seated = false;
  let latest = null;
  let pendingJoin = null;
  let toastTimer = null;
  // Host spectator view: the latest state per watched seat, and the bot panels.
  let spectate = {};
  let panels = {};
  // Move id the host has already confirmed, so the button stays disabled
  // until that turn ends.
  let confirmedMove = null;
  // Last move accounted for, and the game whose fill is on screen.
  // A later move in that game keeps the fill until its first score arrives.
  let barMove = null;
  let barSeed = null;
  const motion = { prev: null, token: 0, dealToken: 0, flights: [], dealSig: null };

  nameInput.value = localStorage.getItem(nameKey) || "";

  function esc(value) {
    return String(value).replace(/[&<>"']/g, function (ch) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[ch];
    });
  }

  function pct(value) {
    return value == null ? "–" : Math.round(value * 100) + "%";
  }

  function showToast(message) {
    toast.textContent = message;
    toast.classList.remove("hidden");
    clearTimeout(toastTimer);
    toastTimer = setTimeout(function () { toast.classList.add("hidden"); }, 3200);
  }

  function send(message) {
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify(message));
    }
  }

  function sendAction(action) {
    closePicker();
    send({ type: "action", action: action, move: moveId });
  }

  function scoreText(msg) {
    const names = msg.names || [];
    const wins = (msg.score && msg.score.wins) || [0, 0];
    const left = names[0] || "Player 1";
    const right = names[1] || "Player 2";
    let text = left + " " + wins[0] + " – " + wins[1] + " " + right;
    const draws = msg.score ? msg.score.draws : 0;
    if (draws) text += " · " + draws + (draws === 1 ? " draw" : " draws");
    return text;
  }

  function otherSeat(msg) {
    return msg.seat === 0 ? 1 : 0;
  }

  function seatName(msg, seat) {
    return (msg.names && msg.names[seat]) || "Player " + (seat + 1);
  }

  function renderHostBar(msg) {
    if (!msg.host) {
      hostBar.classList.add("hidden");
      reviewLink.classList.add("hidden");
      return;
    }
    hostBar.classList.remove("hidden");
    reviewLink.classList.remove("hidden");
    modeButtons.replaceChildren();
    (msg.modes || []).forEach(function (mode) {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "mode-button" + (mode.id === msg.mode ? " active" : "");
      button.textContent = mode.name;
      button.disabled = mode.id !== 1 && !msg.bot_available;
      button.addEventListener("click", function () {
        if (mode.id === msg.mode) return;
        if (msg.phase === "playing" &&
            !window.confirm("Switch to \"" + mode.name + "\"? The current game ends and the score starts again.")) {
          return;
        }
        send({ type: "set_mode", mode: mode.id });
      });
      modeButtons.appendChild(button);
    });
    botNote.textContent = msg.bot_note || "";
    reviewLink.href = "/review?host=" + encodeURIComponent(hostToken);
  }

  function resignApplies(msg) {
    if (!msg || msg.mode === 4) return false;
    if (msg.spectating) return msg.mode === 3;
    return msg.seat === 0 || msg.seat === 1;
  }

  function syncTableActions(msg) {
    const dealt = msg.phase === "playing" || msg.phase === "over";
    resetScoreButton.classList.toggle("hidden", !(msg.host && dealt));
    resignButton.classList.toggle("hidden", !resignApplies(msg));
    resignButton.disabled = msg.phase !== "playing";
  }

  function resignDetail(msg, result, oppLabel) {
    if (msg.spectating) {
      const loser = result.winner === 0 ? 1 : 0;
      return seatName(msg, loser) + " resigned.";
    }
    return result.winner === msg.seat ? oppLabel + " resigned." : "You resigned.";
  }

  function rectOf(el) {
    if (!el || !el.isConnected) return null;
    const rect = el.getBoundingClientRect();
    if (rect.width < 2 || rect.height < 2) return null;
    return { left: rect.left, top: rect.top, width: rect.width, height: rect.height };
  }

  function capturePlaces() {
    const hand = {};
    const oppIds = {};
    handEl.querySelectorAll("[data-id]").forEach(function (el) {
      const rect = rectOf(el);
      if (rect) hand[el.dataset.id] = rect;
    });
    oppCards.querySelectorAll("[data-id]").forEach(function (el) {
      const rect = rectOf(el);
      if (rect) oppIds[el.dataset.id] = rect;
    });
    const unders = [];
    discardUnderEl.querySelectorAll(".card").forEach(function (el) {
      const rect = rectOf(el);
      if (!rect) return;
      unders.push({ rect: rect, suit: el.dataset.suit, rank: el.dataset.rank });
    });
    const oppCard = oppCards.querySelector(".card:last-child");
    return {
      hand: hand,
      oppIds: oppIds,
      oppCard: rectOf(oppCard),
      opp: rectOf(oppCards),
      discard: rectOf(discardEl),
      draw: rectOf(drawPileEl),
      unders: unders
    };
  }

  function clearSettling() {
    document.querySelectorAll(".settling").forEach(function (el) {
      el.classList.remove("settling");
      delete el.dataset.settle;
    });
  }

  function moveMs() {
    const raw = getComputedStyle(document.documentElement).getPropertyValue("--move").trim();
    const n = parseFloat(raw);
    if (!(n > 0)) return 200;
    return raw.endsWith("s") && !raw.endsWith("ms") ? n * 1000 : n;
  }

  // Where the real card lives now. A redraw replaces the node; the id still finds it.
  function settleOf(flight) {
    if (!flight || !flight.root) return null;
    if (flight.root === "discard") return discardEl.isConnected ? discardEl : null;
    const parent = flight.root === "hand" ? handEl : flight.root === "opp" ? oppCards : discardUnderEl;
    if (flight.id) return parent.querySelector('[data-id="' + flight.id + '"]');
    if (flight.index == null) return null;
    return parent.querySelectorAll(".card")[flight.index] || null;
  }

  function hold(el, token) {
    if (!el) return;
    el.dataset.settle = token;
    el.classList.add("settling");
  }

  function release(el, token) {
    if (!el || !el.isConnected || el.dataset.settle !== token) return;
    el.classList.remove("settling");
    delete el.dataset.settle;
  }

  function releasePark(el) {
    if (!el || el.dataset.settle !== "park") return;
    el.classList.remove("settling");
    delete el.dataset.settle;
  }

  // Drop clones already in the air. Do not reveal: the caller decides what is visible.
  function dropFlights() {
    const flights = motion.flights;
    motion.flights = [];
    motion.dealSig = null;
    flights.forEach(function (flight) {
      flight.done = true;
      flight.timers.forEach(function (timer) { window.clearTimeout(timer); });
      flight.timers = [];
      if (flight.node && flight.node.parentNode) flight.node.remove();
      flight.node = null;
    });
  }

  function restoreHolds() {
    motion.flights.forEach(function (flight) {
      if (flight.done || !flight.root) return;
      hold(settleOf(flight), flight.token);
    });
  }

  function fly(from, to, node, delay, layer, settle) {
    if (!from || !to || !node || !from.width || !to.width) return;
    node.classList.add("flyer");
    if (layer) node.classList.add("flyer-" + layer);
    node.classList.remove("settling", "under-0", "under-1");
    delete node.dataset.settle;
    node.style.left = from.left + "px";
    node.style.top = from.top + "px";
    node.style.width = from.width + "px";
    node.style.height = from.height + "px";
    const arc = document.createElement("div");
    arc.className = "arc";
    while (node.firstChild) arc.appendChild(node.firstChild);
    node.appendChild(arc);
    const dx = to.left - from.left;
    const dy = to.top - from.top;
    const sx = to.width / from.width;
    const sy = to.height / from.height;
    const token = String(++motion.token);
    const flight = {
      token: token,
      root: settle ? settle.root : null,
      id: settle && settle.id != null && settle.id !== "" ? String(settle.id) : null,
      index: settle && settle.index != null ? settle.index : null,
      node: null,
      done: false,
      timers: []
    };
    motion.flights.push(flight);
    hold(settleOf(flight), token);
    const finish = function () {
      if (flight.done) return;
      flight.done = true;
      flight.timers.forEach(function (timer) { window.clearTimeout(timer); });
      flight.timers = [];
      if (flight.node && flight.node.parentNode) flight.node.remove();
      flight.node = null;
      release(settleOf(flight), token);
      motion.flights = motion.flights.filter(function (item) { return item !== flight; });
    };
    const start = function () {
      if (flight.done) return;
      flight.node = node;
      document.body.appendChild(node);
      requestAnimationFrame(function () {
        requestAnimationFrame(function () {
          if (flight.done || !node.isConnected) return;
          node.addEventListener("transitionend", function (event) {
            if (event.target === node && (event.propertyName === "transform" || event.propertyName === "-webkit-transform")) {
              finish();
            }
          });
          flight.timers.push(window.setTimeout(finish, moveMs() + 50));
          node.style.transform = "translate(" + dx + "px," + dy + "px) scale(" + sx + "," + sy + ")";
        });
      });
    };
    if (delay) flight.timers.push(window.setTimeout(start, delay));
    else start();
  }

  function nudge(el, className, delay) {
    if (!el) return;
    window.setTimeout(function () {
      el.classList.remove(className);
      void el.offsetWidth;
      el.classList.add(className);
      window.setTimeout(function () { el.classList.remove(className); }, 200);
    }, delay || 0);
  }

  function tagCard(el, card) {
    el.dataset.id = String(card.id);
    el.dataset.suit = card.suit;
    el.dataset.rank = card.rank;
    return el;
  }

  function faceNode(card) {
    return tagCard(prsiCards.element(card), card);
  }

  function isDeal(prev, events, msg) {
    if (events.length !== 1 || !events[0] || events[0].kind !== "opening") return false;
    if (!prev) return true;
    if (prev.phase === "waiting" || prev.phase === "over") return true;
    // A reset stays in "playing" and the new log is one opening line, so the
    // length does not shrink when the old hand had only that line too.
    if (msg.seed != null && prev.seed != null && msg.seed !== prev.seed) return true;
    return events.length < prev.logLen;
  }

  function cancelDeal() {
    motion.dealToken += 1;
    dropFlights();
  }

  function scheduleDeal() {
    const token = ++motion.dealToken;
    requestAnimationFrame(function () {
      if (token !== motion.dealToken) return;
      deal();
    });
  }

  function dealTargets() {
    const targets = [];
    oppCards.querySelectorAll(".card").forEach(function (el, index) {
      targets.push({ el: el, root: "opp", id: el.dataset.id || null, index: index });
    });
    handEl.querySelectorAll(".card").forEach(function (el) {
      targets.push({ el: el, root: "hand", id: el.dataset.id || null, index: null });
    });
    if (discardEl.querySelector("img")) {
      targets.push({ el: discardEl, root: "discard", id: null, index: null });
    }
    return targets;
  }

  function dealSignature() {
    const opp = [];
    const hand = [];
    oppCards.querySelectorAll(".card").forEach(function (el) {
      opp.push(el.dataset.id || "back");
    });
    handEl.querySelectorAll(".card").forEach(function (el) {
      hand.push(el.dataset.id || "");
    });
    const top = discardEl.querySelector("img") ? (discardEl.dataset.id || "") : "";
    return opp.join(",") + "|" + hand.join(",") + "|" + top;
  }

  function parkDealTargets() {
    dealTargets().forEach(function (target) {
      target.el.dataset.settle = "park";
      target.el.classList.add("settling");
    });
  }

  function deal() {
    const targets = dealTargets();
    const draw = rectOf(drawPileEl);
    if (!draw) {
      targets.forEach(function (target) { releasePark(target.el); });
      return;
    }
    motion.dealSig = dealSignature();
    targets.forEach(function (target, index) {
      fly(draw, rectOf(target.el), target.el.cloneNode(true), index * 55, null, {
        root: target.root,
        id: target.id,
        index: target.index
      });
      releasePark(target.el);
    });
  }

  function playEvent(event, delay, prev, places, msg) {
    const mine = event.player === msg.seat;
    if (event.kind === "play" && event.card) {
      const card = event.card;
      const to = rectOf(discardEl);
      const from = mine
        ? (places.hand[String(card.id)] || places.discard)
        : ((places.oppIds && places.oppIds[String(card.id)]) || places.oppCard || places.opp);
      fly(from, to, faceNode(card), delay, "over", { root: "discard" });
      if (prev.topId != null) {
        const underEl = discardUnderEl.querySelector('[data-id="' + prev.topId + '"]');
        const underTo = rectOf(underEl);
        if (underEl && places.discard && underTo) {
          fly(places.discard, underTo, underEl.cloneNode(true), delay, "under", {
            root: "under",
            id: underEl.dataset.id
          });
        }
      }
      return;
    }
    if (event.kind === "draw") {
      const count = event.count || 1;
      if (mine) {
        const known = {};
        (prev.handIds || []).forEach(function (id) { known[id] = true; });
        const fresh = [];
        handEl.querySelectorAll("[data-id]").forEach(function (el) {
          if (!known[el.dataset.id]) fresh.push(el);
        });
        fresh.forEach(function (el, index) {
          const step = delay + index * 40;
          fly(places.draw, rectOf(el), el.cloneNode(true), step, null, {
            root: "hand",
            id: el.dataset.id
          });
        });
      } else if (places.draw) {
        const backs = oppCards.querySelectorAll(".card");
        const target = backs.length ? backs[backs.length - 1] : null;
        const dest = rectOf(target) || rectOf(oppCards);
        for (let i = 0; i < count; i += 1) {
          fly(places.draw, dest, prsiCards.backElement(), delay + i * 40);
        }
      }
      return;
    }
    if (event.kind === "stand" || event.kind === "pass") {
      nudge(mine ? handEl : oppCards, "dip", delay);
      return;
    }
    if (event.kind === "shuffle") {
      const draw = rectOf(drawPileEl);
      (places.unders || []).forEach(function (under, index) {
        if (!under.suit) {
          fly(under.rect, draw, prsiCards.backElement(), delay + index * 30);
          return;
        }
        fly(under.rect, draw, faceNode({
          id: "under",
          suit: under.suit,
          rank: under.rank,
          name: ""
        }), delay + index * 30);
      });
      nudge(drawPileEl, "riffle", delay);
    }
  }

  function runMotion(prev, places, msg, view) {
    const events = view.log || [];
    // The other seat's copy of this position. Rebuilding the cards must not
    // cancel a deal that is waiting for the next frame. Clones already in
    // the air keep flying; the new cards stay hidden until those flights end.
    // A changed card list (backs, then faces) starts the deal over.
    if (prev && msg.seed != null && prev.seed != null && msg.seed === prev.seed &&
        msg.phase === prev.phase && events.length === prev.logLen) {
      if (motion.dealSig && motion.flights.length && dealSignature() !== motion.dealSig) {
        cancelDeal();
        parkDealTargets();
        scheduleDeal();
        return;
      }
      restoreHolds();
      return;
    }
    clearSettling();
    cancelDeal();
    if (isDeal(prev, events, msg)) {
      scheduleDeal();
      return;
    }
    if (!prev || prev.phase !== "playing" || !places) return;
    if (events.length < prev.logLen) return;
    const added = events.slice(prev.logLen);
    if (!added.length || added.length > 2) return;
    added.forEach(function (event, index) {
      playEvent(event, index * 80, prev, places, msg);
    });
  }

  function remember(msg, view) {
    motion.prev = {
      phase: msg.phase,
      seed: msg.seed != null ? msg.seed : null,
      logLen: (view && view.log) ? view.log.length : 0,
      handIds: ((view && view.hand) || []).map(function (card) { return String(card.id); }),
      topId: view && view.top_card ? String(view.top_card.id) : null
    };
  }

  function render(msg) {
    const places = board.classList.contains("hidden") ? null : capturePlaces();
    const prev = motion.prev;
    latest = msg;
    seated = true;
    joinScreen.classList.add("hidden");
    table.classList.remove("hidden");
    roomLabel.textContent = msg.code;
    scoreEl.textContent = scoreText(msg);
    renderHostBar(msg);
    syncTableActions(msg);
    const watchingSeat = Boolean(msg.spectating);
    const opp = otherSeat(msg);
    const oppLabel = seatName(msg, opp);
    const youLabel = watchingSeat ? seatName(msg, msg.seat) : ((msg.names && msg.names[msg.seat]) || "You");

    if (msg.phase === "waiting") {
      table.classList.remove("is-your-turn");
      turnLine.textContent = "";
      turnLine.classList.remove("mine");
      waiting.classList.remove("hidden");
      board.classList.add("hidden");
      resultEl.classList.add("hidden");
      waitingCode.textContent = msg.code;
      waitingText.textContent = msg.notice || "Waiting for your friend to join.";
      waitingInvite.classList.toggle("hidden", Boolean(msg.notice));
      remember(msg, null);
      return;
    }

    waiting.classList.add("hidden");
    board.classList.remove("hidden");
    const view = msg.view;
    moveId = msg.move;

    oppName.textContent = oppLabel;
    const oppCount = view.opponent_hand_size;
    oppMeta.textContent = oppCount + (oppCount === 1 ? " card" : " cards");
    if (msg.connected && msg.connected[opp] === false) {
      oppMeta.textContent += " · disconnected";
    }
    const oppScroll = oppCards.scrollLeft;
    oppCards.replaceChildren();
    const otherSide = watchingSeat && spectate[opp] ? spectate[opp].view : null;
    if (otherSide && otherSide.hand) {
      oppCards.classList.add("open");
      otherSide.hand.forEach(function (card) {
        const face = prsiCards.element(card);
        tagCard(face, card);
        oppCards.appendChild(face);
      });
    } else {
      oppCards.classList.remove("open");
      // A wide face-up hand pins this same row to the left so it can scroll.
      // That pin would send the face-down deal to the edge of the table.
      oppCards.style.justifyContent = "";
      const shown = Math.min(oppCount, 6);
      for (let i = 0; i < shown; i += 1) oppCards.appendChild(prsiCards.backElement());
    }
    oppCards.scrollLeft = oppScroll;

    drawCount.textContent = String(view.draw_pile_size);
    discardUnderEl.replaceChildren();
    (view.under_cards || []).forEach(function (card, index) {
      const under = prsiCards.element(card);
      under.classList.add("under-" + index);
      tagCard(under, card);
      discardUnderEl.appendChild(under);
    });
    if (view.top_card) {
      prsiCards.paint(discardEl, view.top_card, {
        ribbon: view.named_by_over ? view.active_suit_name : "",
        ribbonSuit: view.active_suit
      });
      tagCard(discardEl, view.top_card);
    }

    const yourTurn = Boolean(view.your_turn) && !watchingSeat;
    table.classList.toggle("is-your-turn", msg.phase === "playing" && Boolean(view.your_turn));
    if (watchingSeat) {
      turnLine.textContent = view.current == null || view.result ? "" : seatName(msg, view.current) + "'s turn";
      turnLine.classList.remove("mine");
    } else {
      turnLine.textContent = yourTurn ? "Your turn" : oppLabel + "'s turn";
      turnLine.classList.toggle("mine", yourTurn);
    }

    const legal = new Set(yourTurn ? (view.legal_actions || []) : []);
    handEl.classList.toggle("watching", watchingSeat);
    const handScroll = handEl.scrollLeft;
    handEl.replaceChildren();
    (view.hand || []).forEach(function (card) {
      const plays = (card.actions || []).filter(function (play) { return legal.has(play.id); });
      const button = document.createElement("button");
      button.type = "button";
      prsiCards.paint(button, card);
      tagCard(button, card);
      button.setAttribute("aria-disabled", plays.length ? "false" : "true");
      button.addEventListener("click", function () {
        if (!plays.length) {
          nudge(button, "refuse", 0);
          return;
        }
        onCard(card, plays);
      });
      handEl.appendChild(button);
    });
    handEl.scrollLeft = handScroll;
    youName.textContent = youLabel + (watchingSeat ? " · you are watching" : "");
    const mineCount = (view.hand || []).length;
    youMeta.textContent = mineCount + (mineCount === 1 ? " card" : " cards");

    const pass = view.dont_play || {};
    if (yourTurn && pass.legal) {
      passButton.textContent = pass.label;
      passButton.classList.remove("hidden");
      passButton.onclick = function () { sendAction(pass.id); };
    } else {
      passButton.classList.add("hidden");
      passButton.onclick = null;
    }

    const events = view.log || [];
    const fromBottom = logEl.scrollHeight - logEl.scrollTop - logEl.clientHeight;
    const stick = fromBottom < 40;
    const kept = logEl.scrollTop;
    logEl.replaceChildren();
    events.forEach(function (event) {
      const item = document.createElement("li");
      if (!prsiCards.formatEvent(item, event, msg.seat, watchingSeat ? msg.names : null, oppLabel)) return;
      const suit = prsiCards.lineSuit(event);
      if (suit) item.className = "suit-" + suit;
      logEl.appendChild(item);
    });
    logEl.scrollTop = stick ? logEl.scrollHeight : kept;

    if (msg.phase === "over" && view.result) {
      const history = view.log || [];
      const added = prev && prev.phase === "playing" ? history.slice(prev.logLen) : [];
      const playedOut = added.length > 0 && added.length <= 2 && added.some(function (event) {
        return event.kind === "play";
      });
      if (playedOut) {
        resultEl.classList.add("hidden");
        window.setTimeout(function () {
          if (latest === msg) showResult(msg, view, oppLabel);
        }, 240);
      } else {
        showResult(msg, view, oppLabel);
      }
    } else {
      resultEl.classList.add("hidden");
    }
    renderPanels();
    syncLiveBar(msg);
    runMotion(prev, places, msg, view);
    remember(msg, view);
    scheduleFit();
  }

  function showResult(msg, view, oppLabel) {
    const result = view.result;
    if (msg.spectating) {
      let title = "It's a draw";
      if (result.winner != null) title = seatName(msg, result.winner) + " wins";
      resultTitle.textContent = title;
      resultDetail.textContent = result.reason === "resign"
        ? resignDetail(msg, result, seatName(msg, result.winner === 0 ? 1 : 0))
        : result.reason === "empty_hand"
        ? "The last card was played."
        : "Nothing left to draw, and neither player could play.";
      resultScore.textContent = scoreText(msg);
      if (msg.mode === 4) {
        rematchButton.classList.remove("hidden");
        rematchButton.disabled = false;
        rematchButton.textContent = "Next game";
        rematchNote.textContent = "";
      } else {
        rematchButton.classList.add("hidden");
        rematchNote.textContent = "Your friend decides whether to play again.";
      }
      resultEl.classList.remove("hidden");
      return;
    }
    rematchButton.classList.remove("hidden");
    let title = "It's a draw";
    let detail = "Nothing left to draw, and neither player can play. You have the same number of cards.";
    if (result.reason === "resign") {
      if (result.winner === msg.seat) {
        title = "You win";
        detail = resignDetail(msg, result, oppLabel);
      } else if (result.winner != null) {
        title = "You lose";
        detail = resignDetail(msg, result, oppLabel);
      }
    } else if (result.winner === msg.seat) {
      title = "You win";
      detail = result.reason === "empty_hand"
        ? "You played your last card."
        : "Nothing left to draw, and neither player can play. You have fewer cards.";
    } else if (result.winner != null) {
      title = "You lose";
      detail = result.reason === "empty_hand"
        ? oppLabel + " played their last card."
        : "Nothing left to draw, and neither player can play. " + oppLabel + " has fewer cards.";
    }
    resultTitle.textContent = title;
    resultDetail.textContent = detail;
    resultScore.textContent = scoreText(msg);
    const mine = msg.rematch && msg.rematch[msg.seat];
    const theirs = msg.rematch && msg.rematch[otherSeat(msg)];
    rematchButton.disabled = Boolean(mine);
    rematchButton.textContent = mine ? "Waiting for opponent…" : "Rematch";
    rematchNote.textContent = !mine && theirs ? oppLabel + " wants a rematch." : "";
    resultEl.classList.remove("hidden");
  }

  function botNeedsConfirm() {
    return Boolean(
      latest && latest.mode === 3 && latest.spectating && latest.phase === "playing" &&
      latest.view && latest.view.your_turn
    );
  }

  function confirmMove() {
    if (!latest || confirmedMove === latest.move) return;
    confirmedMove = latest.move;
    const button = document.getElementById("confirm-move");
    if (button) button.disabled = true;
    send({ type: "confirm_move", move: latest.move });
  }

  function placeTurnGlow() {
    const zone = document.querySelector(".turn-zone");
    if (!zone) return;
    const span = Math.max(0, window.innerHeight - zone.getBoundingClientRect().top);
    zone.style.setProperty("--glow-span", span + "px");
  }

  function scheduleFit() {
    requestAnimationFrame(function () {
      window.prsiCards.fit(table);
      placeTurnGlow();
    });
  }

  function syncConfirmButton() {
    // One button, fixed in the rail, so a refresh of the lines cannot move it.
    const button = document.getElementById("confirm-move");
    if (!button) return;
    const show = botNeedsConfirm();
    button.classList.toggle("hidden", !show);
    if (show) button.disabled = confirmedMove === latest.move;
  }

  function lineLimit(input) {
    const count = Math.floor(Number(input && input.value));
    return count >= 1 ? count : 3;
  }

  function renderPanels() {
    if (!latest || !latest.spectating || latest.phase === "waiting") {
      panelsFold.classList.add("hidden");
      panelsEl.replaceChildren();
      playStage.classList.remove("has-lines");
      syncConfirmButton();
      return;
    }
    const seats = (latest.watching || [latest.seat]).slice();
    // The table draws the higher seat on top. List that bot first.
    if (latest.mode === 4) seats.sort(function (a, b) { return b - a; });
    const keep = {};
    seats.forEach(function (seat) { keep[String(seat)] = true; });
    panelsEl.querySelectorAll(".panel-box").forEach(function (box) {
      if (!keep[box.dataset.seat]) box.remove();
    });
    seats.forEach(function (seat) {
      const panel = panels[seat];
      let box = panelsEl.querySelector('.panel-box[data-seat="' + seat + '"]');
      if (!box) {
        box = document.createElement("div");
        box.className = "panel-box";
        box.dataset.seat = String(seat);
        const body = document.createElement("div");
        body.className = "panel-body";
        box.appendChild(body);
      }
      panelsEl.appendChild(box);
      const name = seatName(latest, seat);
      const unseenOpen = Boolean(box.querySelector(".unseen-fold") && box.querySelector(".unseen-fold").open);
      let html;
      if (!panel) {
        html = "<h2>" + esc(name) + "</h2><p class=\"hint\">The evaluation appears when this bot thinks.</p>";
      } else {
        const pace = panel.paused
          ? " · <em>waiting for confirmation</em>"
          : (panel.final ? "" : " · <em>thinking…</em>");
        html = "<div class=\"panel-head\"><h2>" + esc(name) + "</h2>" +
          "<p><span class=\"win\">" + pct(panel.win_chance) + "</span> win chance<br>" +
          Number(panel.iterations || 0).toLocaleString() + " imagined games" +
          pace + "</p></div>";
        html += "<table class=\"moves\"><thead><tr><th>Move</th><th>Win chance</th><th>Tried</th><th>Network</th></tr></thead><tbody>";
        panel.moves.slice(0, lineLimit(topLinesInput)).forEach(function (move) {
          const chosen = move.action === panel.action;
          html += "<tr class=\"" + (chosen ? "chosen" : "") + "\"><td>" + esc(move.label) + "</td><td>" + pct(move.win_chance) +
            "</td><td>" + Number(move.visits || 0).toLocaleString() + "</td><td>" + pct(move.prior) + "</td></tr>";
        });
        html += "</tbody></table>";
        html += "<details class=\"unseen-fold\"" + (unseenOpen ? " open" : "") + ">" +
          "<summary>Unseen cards (" + panel.unseen.length + ")</summary>" +
          "<p class=\"unseen-title\">" + panel.opponent_cards + " in the opponent's hand, " +
          panel.draw_pile + " in the draw pile. <b>Bold</b> = certainly in the opponent's hand.</p><p class=\"unseen\">";
        panel.unseen.forEach(function (card) {
          html += "<span class=\"chip suit-" + card.suit + (card.certain ? " certain" : "") + "\">" + esc(card.name) + "</span>";
        });
        html += "</p></details>";
      }
      box.querySelector(".panel-body").innerHTML = html;
    });
    syncConfirmButton();
    panelsFold.classList.remove("hidden");
    playStage.classList.add("has-lines");
    scheduleFit();
  }

  function onCard(card, plays) {
    if (!plays.length) return;
    if (plays.length === 1) {
      sendAction(plays[0].id);
      return;
    }
    pickerCard.textContent = card.name;
    pickerSuits.replaceChildren();
    plays.forEach(function (play) {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "suit-choice suit-" + play.suit;
      button.textContent = play.suit_name;
      button.addEventListener("click", function () { sendAction(play.id); });
      pickerSuits.appendChild(button);
    });
    picker.classList.remove("hidden");
  }

  function closePicker() { picker.classList.add("hidden"); }

  // A scored bar glides. Empty, and the first score after empty, snap:
  // the transition is turned off, the new grow is committed, then it is armed again.
  function placeFill(bar, topEl, bottomEl, topGrow, bottomGrow, glide) {
    if (!glide) bar.classList.add("no-glide");
    topEl.style.flexGrow = String(topGrow);
    bottomEl.style.flexGrow = String(bottomGrow);
    if (!glide) {
      void bar.offsetWidth;
      bar.classList.remove("no-glide");
    }
  }

  function clearLiveBar() {
    placeFill(liveEvalBar, liveEvalTop, liveEvalBottom, 1, 1, false);
    liveEvalBar.classList.add("empty");
    liveEvalPct.textContent = "–";
    liveEvalPct.classList.add("is-empty");
    liveEvalCount.textContent = "";
    liveEvalBar.setAttribute("aria-label", "Win chance not scored yet");
    barSeed = null;
  }

  function paintLiveBar(bar) {
    const chance = bar && typeof bar === "object" ? bar.chance : bar;
    const bottom = Math.max(0, Math.min(1, Number(chance)));
    const glide = !liveEvalBar.classList.contains("empty");
    placeFill(liveEvalBar, liveEvalTop, liveEvalBottom, 1 - bottom, bottom, glide);
    liveEvalBar.classList.remove("empty");
    liveEvalPct.textContent = pct(bottom);
    liveEvalPct.classList.remove("is-empty");
    const who = (latest && latest.names && latest.names[0]) || "Bottom player";
    const samples = bar && bar.samples;
    const cap = bar && bar.cap;
    if (samples && cap) {
      liveEvalCount.textContent = samples + " / " + cap;
      liveEvalBar.setAttribute(
        "aria-label",
        who + " win chance " + pct(bottom) + ", " + samples + " of " + cap + " finishes"
      );
    } else {
      liveEvalCount.textContent = "";
      liveEvalBar.setAttribute("aria-label", who + " win chance " + pct(bottom));
    }
    if (latest && latest.seed != null) barSeed = latest.seed;
  }

  function syncLiveBar(msg) {
    const show = msg.mode === 4 && msg.spectating && msg.phase !== "waiting";
    liveEval.classList.toggle("hidden", !show);
    if (!show) {
      barMove = null;
      return;
    }
    const names = msg.names || ["Bot A", "Bot B"];
    liveEvalName0.textContent = names[0] || "Bot A";
    liveEvalName1.textContent = names[1] || "Bot B";
    if (msg.bar && msg.bar.move === msg.move) {
      paintLiveBar(msg.bar);
      barMove = msg.move;
      return;
    }
    if (msg.move !== barMove) {
      const sameGame = barSeed != null && msg.seed != null && msg.seed === barSeed &&
        !liveEvalBar.classList.contains("empty");
      barMove = msg.move;
      if (!sameGame) clearLiveBar();
    }
  }

  function onState(msg) {
    if (msg.spectating) {
      const watched = msg.watching || [msg.seat];
      Object.keys(spectate).forEach(function (seat) {
        if (watched.indexOf(Number(seat)) < 0) delete spectate[seat];
      });
      spectate[msg.seat] = msg;
      panels[msg.seat] = msg.panel || null;
      // Bot vs bot: seat 0 is drawn at the bottom, seat 1 at the top.
      const primary = spectate[Math.min.apply(null, watched)] || msg;
      render(primary);
    } else {
      spectate = {};
      panels = {};
      render(msg);
    }
  }

  function connect() {
    if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) return;
    const protocol = location.protocol === "https:" ? "wss" : "ws";
    ws = new WebSocket(protocol + "://" + location.host + "/ws");
    linkStatus.textContent = "Connecting…";
    linkStatus.classList.remove("live");
    ws.onopen = function () {
      retryMs = 1000;
      linkStatus.textContent = "Connected";
      linkStatus.classList.add("live");
      const token = localStorage.getItem(storageKey);
      if (token) send({ type: "rejoin", token: token });
      else if (pendingJoin) send(pendingJoin);
    };
    ws.onmessage = function (event) {
      let msg;
      try { msg = JSON.parse(event.data); } catch (err) { return; }
      if (msg.type === "joined") {
        pendingJoin = null;
        localStorage.setItem(storageKey, msg.token);
        seated = true;
        joinError.textContent = "";
        return;
      }
      if (msg.type === "error") {
        if (!seated) {
          pendingJoin = null;
          lastSeq = 0;
          joinError.textContent = msg.message;
          localStorage.removeItem(storageKey);
          joinScreen.classList.remove("hidden");
          table.classList.add("hidden");
        } else {
          if (confirmedMove === moveId) {
            confirmedMove = null;
            renderPanels();
          }
          showToast(msg.message);
        }
        return;
      }
      if (msg.type === "toast") {
        showToast(msg.message);
        return;
      }
      if (msg.type === "panel") {
        panels[msg.seat] = msg.panel;
        renderPanels();
        return;
      }
      if (msg.type === "bar") {
        if (latest && latest.mode === 4 && msg.move === latest.move) {
          paintLiveBar(msg);
          barMove = msg.move;
        }
        return;
      }
      if (msg.type === "state") {
        if (typeof msg.seq === "number" && msg.seq < lastSeq) return;
        if (typeof msg.seq === "number") lastSeq = msg.seq;
        onState(msg);
      }
    };
    ws.onclose = function () {
      linkStatus.textContent = "Reconnecting…";
      linkStatus.classList.remove("live");
      clearTimeout(retryTimer);
      retryTimer = setTimeout(connect, retryMs);
      retryMs = Math.min(retryMs * 1.5, 5000);
    };
  }

  joinForm.addEventListener("submit", function (event) {
    event.preventDefault();
    const name = nameInput.value.trim();
    const code = codeInput.value.trim();
    if (!name || !code) return;
    localStorage.setItem(nameKey, name);
    joinError.textContent = "";
    seated = false;
    lastSeq = 0;
    localStorage.removeItem(storageKey);
    pendingJoin = { type: "join", code: code, name: name, host: hostToken };
    if (!ws || ws.readyState === WebSocket.CLOSED) connect();
    else if (ws.readyState === WebSocket.OPEN) send(pendingJoin);
  });

  document.getElementById("copy-code").addEventListener("click", function () {
    const code = waitingCode.textContent || "";
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(code).then(function () {
        showToast("Room code copied.");
      }, function () { showToast(code); });
    } else {
      showToast(code);
    }
  });

  rematchButton.addEventListener("click", function () { send({ type: "rematch" }); });
  resetScoreButton.addEventListener("click", function () {
    if (!latest || !latest.host) return;
    const question = latest.phase === "playing"
      ? "Reset the score to 0–0 and deal a new hand? This hand will not count."
      : "Reset the score to 0–0 and deal a new hand?";
    if (!window.confirm(question)) return;
    send({ type: "reset_score" });
  });
  resignButton.addEventListener("click", function () {
    if (!latest || latest.phase !== "playing") return;
    const question = latest.spectating && latest.mode === 3
      ? "Resign this hand? Your friend wins."
      : "Resign this hand? Your opponent wins.";
    if (!window.confirm(question)) return;
    send({ type: "resign" });
  });
  document.getElementById("confirm-move").addEventListener("click", confirmMove);
  document.getElementById("picker-cancel").addEventListener("click", closePicker);
  picker.addEventListener("click", function (event) { if (event.target === picker) closePicker(); });
  document.getElementById("join-help").addEventListener("click", function () { help.classList.remove("hidden"); });
  document.getElementById("help-button").addEventListener("click", function () { help.classList.remove("hidden"); });
  document.getElementById("help-close").addEventListener("click", function () { help.classList.add("hidden"); });
  topLinesInput.addEventListener("input", renderPanels);
  help.addEventListener("click", function (event) { if (event.target === help) help.classList.add("hidden"); });
  document.addEventListener("keydown", function (event) {
    if (event.key === "Escape") {
      closePicker();
      help.classList.add("hidden");
    }
  });

  if (localStorage.getItem(storageKey)) {
    joinError.textContent = "Reconnecting…";
  }
  window.prsiCards.watch(table);
  window.addEventListener("resize", function () {
    requestAnimationFrame(placeTurnGlow);
  });
  connect();
})();
