(function () {
  // Hearts are the pomegranate engravings. The English name stays Hearts.
  var SUIT_FILE = {
    hearts: "pomegranates",
    leaves: "leaves",
    acorns: "acorns",
    bells: "bells"
  };

  function faceUrl(card) {
    var suit = SUIT_FILE[card.suit] || card.suit;
    return "/cards/" + suit + "-" + card.rank + ".png";
  }

  var RANK_FILE = ["7", "8", "9", "10", "under", "over", "king", "ace"];
  var preloaded = [];

  function preloadFaces() {
    var seen = {};
    var urls = ["/cards/back.png"];
    Object.keys(SUIT_FILE).forEach(function (key) {
      var suit = SUIT_FILE[key];
      if (seen[suit]) return;
      seen[suit] = true;
      RANK_FILE.forEach(function (rank) {
        urls.push("/cards/" + suit + "-" + rank + ".png");
      });
    });
    urls.forEach(function (url) {
      var img = new Image();
      img.src = url;
      preloaded.push(img);
      if (img.decode) img.decode().catch(function () {});
    });
  }

  preloadFaces();

  function image(src) {
    var img = document.createElement("img");
    img.src = src;
    img.alt = "";
    img.draggable = false;
    return img;
  }

  function paint(el, card, opts) {
    opts = opts || {};
    el.className = "card suit-" + card.suit;
    if (card.name) el.setAttribute("aria-label", card.name);
    el.replaceChildren(image(faceUrl(card)));
    if (opts.ribbon) {
      var ribbon = document.createElement("span");
      ribbon.className = "ribbon" + (opts.ribbonSuit ? " suit-" + opts.ribbonSuit : "");
      ribbon.textContent = opts.ribbon;
      el.appendChild(ribbon);
    }
    return el;
  }

  function paintBack(el) {
    el.className = "card back";
    el.removeAttribute("aria-label");
    el.replaceChildren(image("/cards/back.png"));
    return el;
  }

  function element(card, opts) {
    var el = document.createElement(opts && opts.button ? "button" : "div");
    if (opts && opts.button) el.type = "button";
    return paint(el, card, opts);
  }

  function backElement() {
    return paintBack(document.createElement("div"));
  }

  function appendCardName(parent, name) {
    var strong = document.createElement("strong");
    strong.className = "card-name";
    strong.textContent = name;
    parent.appendChild(strong);
  }

  // ``names`` is set when watching: then every line uses the seat names.
  // Returns false when the event has no public line.
  function formatEvent(item, event, you, names, oppLabel) {
    var mine = event.player === you;
    var who = names ? (names[event.player] || "Someone") : (mine ? "You" : "Opponent");
    if (event.kind === "opening") {
      item.appendChild(document.createTextNode("Opening card: "));
      appendCardName(item, event.card.name);
      item.appendChild(document.createTextNode("."));
      return true;
    }
    if (event.kind === "play") {
      item.appendChild(document.createTextNode(who + " played "));
      appendCardName(item, event.card.name);
      if (event.suit_name) item.appendChild(document.createTextNode(" and named " + event.suit_name));
      item.appendChild(document.createTextNode("."));
      return true;
    }
    var line = "";
    if (event.kind === "draw") {
      if (event.penalty) {
        var noun = event.count === 1 ? "card" : "cards";
        line = who + " took " + event.count + " " + noun + ".";
      } else if (names) {
        line = who + " drew a card.";
      } else {
        line = mine ? "You drew a card." : "Opponent drew a card.";
      }
    } else if (event.kind === "stand") {
      line = names ? who + " stood." : (mine ? "You stood." : "Opponent stood.");
    } else if (event.kind === "pass") {
      var extra = event.forgiven ? " — no cards left to take" : "";
      line = (names ? who + " passed" : (mine ? "You passed" : "Opponent passed")) + extra + ".";
    } else if (event.kind === "shuffle") {
      line = "The discard pile was shuffled into the draw pile.";
    } else if (event.kind === "game_over") {
      if (event.reason === "reset") {
        line = "Hand stopped. The score was reset.";
      } else if (event.reason === "resign") {
        if (names) {
          var loser = event.winner === 0 ? 1 : 0;
          line = (names[loser] || "Someone") + " resigned.";
        } else if (event.winner === you) {
          line = (oppLabel || "Opponent") + " resigned.";
        } else {
          line = "You resigned.";
        }
      } else if (event.winner == null) {
        line = "The game is a draw.";
      } else if (names) {
        line = (names[event.winner] || "Someone") + " wins.";
      } else {
        line = event.winner === you ? "You win." : "Opponent wins.";
      }
    }
    if (!line) return false;
    item.textContent = line;
    return true;
  }

  // A named Over is the suit that was called. Every other card line is the card's own suit.
  function lineSuit(event) {
    if (event.kind === "opening") return event.card && event.card.suit;
    if (event.kind === "play") return event.suit || (event.card && event.card.suit) || "";
    return "";
  }

  window.prsiCards = {
    faceUrl: faceUrl,
    paint: paint,
    paintBack: paintBack,
    element: element,
    backElement: backElement,
    formatEvent: formatEvent,
    lineSuit: lineSuit,
    fit: fit,
    watch: watch
  };

  // How far a side rail reaches in from the screen edge. A hidden rail
  // reaches nothing. The wider rail decides both sides, so the cards stay
  // on the center of the window and still clear the panel.
  function railReach(rail, stageRect) {
    if (!rail) return 0;
    var style = window.getComputedStyle(rail);
    if (style.display === "none" || style.position !== "absolute") return 0;
    var rect = rail.getBoundingClientRect();
    var left = rect.left;
    var right = rect.right;
    var fold = rail.querySelector(".fold");
    if (fold) {
      var foldRect = fold.getBoundingClientRect();
      if (foldRect.width > 0) {
        left = Math.min(left, foldRect.left);
        right = Math.max(right, foldRect.right);
      }
    }
    var fromLeft = right - stageRect.left;
    var fromRight = stageRect.right - left;
    var onLeft = (left + right) / 2 < (stageRect.left + stageRect.right) / 2;
    return Math.max(0, onLeft ? fromLeft : fromRight);
  }

  function safeWidth(stage) {
    if (window.innerWidth <= 800) return stage.clientWidth;
    var stageRect = stage.getBoundingClientRect();
    var inset = 0;
    stage.querySelectorAll(".rail").forEach(function (rail) {
      var reach = railReach(rail, stageRect);
      if (reach > inset) inset = reach;
    });
    return Math.max(160, stageRect.width - 2 * inset);
  }

  function bindFolds(root) {
    root.querySelectorAll(".fold").forEach(function (button) {
      if (button.dataset.bound || button.dataset.swap) return;
      button.dataset.bound = "1";
      button.addEventListener("click", function () {
        var rail = button.closest(".rail");
        if (!rail) return;
        var collapsed = rail.classList.toggle("is-collapsed");
        button.setAttribute("aria-expanded", collapsed ? "false" : "true");
        var label = collapsed ? button.dataset.shutLabel : button.dataset.openLabel;
        if (label) button.setAttribute("aria-label", label);
        fit(root);
      });
    });
  }

  var SCROLL_ROW = ".hand, .hand-row, .backs.open";

  // Grow the cards to the size of a normal five-card hand. A longer hand scrolls.
  function fit(root) {
    if (!root) return;
    var stage = root.querySelector(".play-stage") || root.querySelector(".review-layout");
    var center = root.querySelector(".play-center");
    var board = center || root.querySelector(".game-view");
    if (!stage || !board) return;
    var narrow = window.innerWidth <= 800;
    var width = narrow ? stage.clientWidth : safeWidth(stage);
    var height = narrow ? board.clientHeight : stage.clientHeight;
    if (width < 80 || height < 120) return;
    var count = 5;
    var gap = narrow ? 8 : 14;
    var byWidth = (width - 40 - gap * (count - 1)) / count;
    var chrome = center ? 168 : 196;
    var byHeight = (height - chrome) / 4.46;
    var card = Math.floor(Math.min(byWidth, byHeight, narrow ? 124 : 176));
    if (!(card > 64)) card = 64;
    root.style.setProperty("--card-w", Math.round(card) + "px");
    root.style.setProperty("--safe", Math.round(width) + "px");
    syncBars(root);
  }

  function thumbBox(row, bar) {
    var inner = bar.clientWidth;
    var ratio = row.scrollWidth > 0 ? row.clientWidth / row.scrollWidth : 1;
    var thumbW = Math.max(18, Math.round(inner * ratio) - 4);
    if (thumbW > inner - 4) thumbW = Math.max(0, inner - 4);
    return { thumbW: thumbW, travel: Math.max(0, inner - thumbW - 4) };
  }

  function rowOverflows(row) {
    var style = window.getComputedStyle(row);
    var gap = parseFloat(style.columnGap || style.gap) || 0;
    var need = parseFloat(style.paddingLeft) + parseFloat(style.paddingRight);
    row.querySelectorAll(":scope > .card").forEach(function (card, index) {
      var margin = parseFloat(window.getComputedStyle(card).marginLeft) || 0;
      if (index) need += gap;
      need += margin + card.offsetWidth;
    });
    return need > row.clientWidth + 1;
  }

  function placeThumb(row, bar) {
    var overflows = rowOverflows(row);
    var align = overflows ? "flex-start" : "";
    if (row.style.justifyContent !== align) row.style.justifyContent = align;
    bar.hidden = !overflows;
    if (!overflows) return;
    var thumb = bar.firstElementChild;
    var box = thumbBox(row, bar);
    var maxScroll = row.scrollWidth - row.clientWidth;
    var pos = maxScroll > 0 ? row.scrollLeft / maxScroll : 0;
    thumb.style.width = box.thumbW + "px";
    thumb.style.left = (2 + pos * box.travel) + "px";
  }

  function bindRow(row) {
    if (row.dataset.handScroll) return;
    row.dataset.handScroll = "1";
    var drag = null;
    var suppressClick = false;
    row.addEventListener("scroll", function () {
      var bar = row.nextElementSibling;
      if (bar && bar.classList.contains("hand-bar")) placeThumb(row, bar);
    });
    row.addEventListener("wheel", function (event) {
      if (row.scrollWidth <= row.clientWidth + 1) return;
      if (Math.abs(event.deltaX) > Math.abs(event.deltaY)) return;
      row.scrollLeft += event.deltaY;
      event.preventDefault();
    }, { passive: false });
    function onMove(event) {
      if (!drag || event.pointerId !== drag.id) return;
      var dx = event.clientX - drag.x;
      if (!drag.moved && Math.abs(dx) < 8) return;
      if (!drag.moved) {
        drag.moved = true;
        row.setPointerCapture(event.pointerId);
      }
      row.scrollLeft = drag.left - dx;
    }
    function endDrag(event) {
      if (!drag || event.pointerId !== drag.id) return;
      if (event.type === "pointerup" && drag.moved) suppressClick = true;
      drag = null;
      document.removeEventListener("pointermove", onMove);
      document.removeEventListener("pointerup", endDrag);
      document.removeEventListener("pointercancel", endDrag);
    }
    row.addEventListener("pointerdown", function (event) {
      if (drag || event.button !== 0) return;
      if (row.scrollWidth <= row.clientWidth + 1) return;
      drag = { x: event.clientX, left: row.scrollLeft, moved: false, id: event.pointerId };
      document.addEventListener("pointermove", onMove);
      document.addEventListener("pointerup", endDrag);
      document.addEventListener("pointercancel", endDrag);
    });
    row.addEventListener("click", function (event) {
      if (!suppressClick) return;
      suppressClick = false;
      event.preventDefault();
      event.stopPropagation();
    }, true);
  }

  function scrollFromPointer(row, bar, clientX, grab) {
    var box = thumbBox(row, bar);
    var maxScroll = row.scrollWidth - row.clientWidth;
    if (maxScroll <= 0 || box.travel <= 0) return;
    var rect = bar.getBoundingClientRect();
    var borderLeft = parseFloat(window.getComputedStyle(bar).borderLeftWidth) || 0;
    var local = clientX - rect.left - borderLeft - grab;
    var pos = (local - 2) / box.travel;
    if (pos < 0) pos = 0;
    if (pos > 1) pos = 1;
    row.scrollLeft = pos * maxScroll;
  }

  function bindBar(row, bar) {
    bindRow(row);
    if (bar.dataset.barDrag) return;
    bar.dataset.barDrag = "1";
    var drag = null;
    function endDrag(event) {
      if (!drag || event.pointerId !== drag.id) return;
      drag = null;
      bar.classList.remove("is-dragging");
    }
    bar.addEventListener("pointerdown", function (event) {
      if (drag || event.button !== 0) return;
      if (row.scrollWidth <= row.clientWidth + 1) return;
      event.preventDefault();
      var thumb = bar.firstElementChild;
      var thumbRect = thumb.getBoundingClientRect();
      var onThumb = event.clientX >= thumbRect.left && event.clientX <= thumbRect.right;
      var grab = onThumb ? event.clientX - thumbRect.left : thumbRect.width / 2;
      drag = { id: event.pointerId, grab: grab };
      bar.classList.add("is-dragging");
      bar.setPointerCapture(event.pointerId);
      scrollFromPointer(row, bar, event.clientX, grab);
    });
    bar.addEventListener("pointermove", function (event) {
      if (!drag || event.pointerId !== drag.id) return;
      scrollFromPointer(row, bar, event.clientX, drag.grab);
    });
    bar.addEventListener("pointerup", endDrag);
    bar.addEventListener("pointercancel", endDrag);
  }

  function ensureBar(row) {
    var bar = row.nextElementSibling;
    if (!bar || !bar.classList.contains("hand-bar")) {
      bar = document.createElement("div");
      bar.className = "hand-bar";
      bar.hidden = true;
      bar.setAttribute("aria-hidden", "true");
      bar.appendChild(document.createElement("i"));
      row.after(bar);
    }
    bindBar(row, bar);
    return bar;
  }

  function syncBars(root) {
    root.querySelectorAll(".hand-bar").forEach(function (bar) {
      var row = bar.previousElementSibling;
      if (!row || !row.matches(SCROLL_ROW)) bar.remove();
    });
    root.querySelectorAll(SCROLL_ROW).forEach(function (row) {
      placeThumb(row, ensureBar(row));
    });
  }

  function watch(root) {
    if (!root || root.dataset.cardFit) return;
    root.dataset.cardFit = "1";
    bindFolds(root);
    if (window.ResizeObserver) {
      var observer = new ResizeObserver(function () { fit(root); });
      var stage = root.querySelector(".play-stage") || root.querySelector(".review-layout");
      if (stage) observer.observe(stage);
      root.querySelectorAll(".rail").forEach(function (rail) { observer.observe(rail); });
    }
    fit(root);
    window.addEventListener("resize", function () { fit(root); });
  }
})();
