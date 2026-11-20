const $ = (id) => document.getElementById(id);

const state = { topics: [], domain: "all", busy: false };

const DOMAINS = {
  ai: "AI & tech",
  cyber: "Cybersecurity",
  jobs: "Jobs",
  behaviour: "Behaviour",
  india: "India",
};

function renderFilters() {
  const box = $("filters");
  box.innerHTML = "";
  const domains = ["all", ...new Set(state.topics.map((t) => t.domain))];
  for (const domain of domains) {
    const button = document.createElement("button");
    button.className = "pill" + (domain === state.domain ? " active" : "");
    button.type = "button";
    button.textContent = domain === "all" ? "All topics" : DOMAINS[domain] || domain;
    button.addEventListener("click", () => {
      state.domain = domain;
      renderFilters();
      showSuggestions($("topic").value);
    });
    box.appendChild(button);
  }
}

function candidates(query) {
  const q = query.trim().toLowerCase();
  return state.topics
    .filter((t) => state.domain === "all" || t.domain === state.domain)
    .filter((t) => !q || t.topic.toLowerCase().includes(q));
}

function hideSuggestions() {
  $("suggestions").hidden = true;
}

function showSuggestions(query) {
  const list = candidates(query).slice(0, 8);
  const exact = list.length === 1 && list[0].topic.toLowerCase() === query.trim().toLowerCase();
  if (!list.length || exact) {
    hideSuggestions();
    return;
  }
  const box = $("suggestions");
  box.innerHTML = "";
  for (const item of list) {
    const button = document.createElement("button");
    button.type = "button";
    button.innerHTML = `<span>${item.topic}</span><span class="dom">${DOMAINS[item.domain] || item.domain}</span>`;
    button.addEventListener("click", () => {
      $("topic").value = item.topic;
      hideSuggestions();
      run(item.topic);
    });
    box.appendChild(button);
  }
  box.hidden = false;
}

function renderChips(container, words) {
  container.innerHTML = "";
  words.forEach((word, index) => {
    const chip = document.createElement("span");
    chip.className = "chip";
    chip.textContent = word;
    chip.style.setProperty("--i", String(index));
    chip.title = "Click to copy";
    chip.addEventListener("click", () => {
      navigator.clipboard?.writeText(word).catch(() => {});
      chip.textContent = word + " ✓";
      setTimeout(() => { chip.textContent = word; }, 650);
    });
    container.appendChild(chip);
  });
}

function renderPlaceholder(container, text) {
  container.innerHTML = `<p class="placeholder">${text}</p>`;
}

function flashError(message) {
  $("meta").textContent = message;
  const input = $("topic");
  input.classList.remove("shake");
  void input.offsetWidth;
  input.classList.add("shake");
  setTimeout(() => input.classList.remove("shake"), 400);
}

/** Clear any retrieval banner from previous runs. */
function clearRetrievalBanner() {
  const existing = document.querySelector(".retrieval-banner");
  if (existing) existing.remove();
}

/** Show a banner explaining that retrieval mode was used. */
function showRetrievalBanner(data) {
  clearRetrievalBanner();
  const banner = document.createElement("div");
  banner.className = "retrieval-banner";

  const confidence = data.confidence || 0;
  const isWeak = confidence < 0.15;
  banner.classList.toggle("weak", isWeak);

  const sourceLabels = (data.sources || [])
    .map((s) => `<span class="source-tag">${s.topic} <small>(${Math.round(s.similarity * 100)}%)</small></span>`)
    .join(" ");

  banner.innerHTML =
    `<span class="retrieval-icon">${isWeak ? "⚠" : "↗"}</span>` +
    `<span class="retrieval-text">` +
    `<strong>"${data.topic}"</strong> is not in the 48-topic set. ` +
    (isWeak
      ? `Low confidence — these keywords may not be relevant.`
      : `Showing keywords from the nearest known topics:`) +
    `</span>` +
    `<div class="source-tags">${sourceLabels}</div>`;

  const arena = document.querySelector(".arena");
  arena.parentNode.insertBefore(banner, arena);
}

async function run(rawTopic) {
  if (state.busy) return;
  const topic = rawTopic.trim();
  if (!topic) {
    flashError("Type a topic first.");
    return;
  }

  hideSuggestions();
  clearRetrievalBanner();
  state.busy = true;
  $("go").disabled = true;
  $("go").textContent = "Thinking";
  $("meta").textContent = "The two models are choosing keywords…";

  try {
    const response = await fetch("/api/debate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ topic }),
    });
    if (!response.ok) {
      const err = await response.json().catch(() => ({}));
      throw new Error(err.detail || `HTTP ${response.status}`);
    }
    const data = await response.json();
    renderChips($("pro"), data.pro);
    renderChips($("con"), data.con);
    $("proCount").textContent = `${data.pro.length} keywords`;
    $("conCount").textContent = `${data.con.length} keywords`;

    if (data.mode === "retrieval") {
      showRetrievalBanner(data);
      $("meta").textContent = `${data.topic} · retrieval fallback`;
      $("timing").textContent = `${data.ms} ms · nearest-topic lookup`;
    } else {
      $("meta").textContent = `${data.topic} · ${DOMAINS[data.domain] || data.domain}`;
      $("timing").textContent = `${data.ms} ms · 19M params each`;
    }
  } catch (error) {
    flashError(error.message || "The local model did not answer. Is the server still running?");
    console.error(error);
  } finally {
    state.busy = false;
    $("go").disabled = false;
    $("go").textContent = "Generate";
  }
}

function pickRandom() {
  const pool = candidates("");
  const item = pool[Math.floor(Math.random() * pool.length)];
  $("topic").value = item.topic;
  run(item.topic);
}

function bind() {
  const input = $("topic");
  input.addEventListener("input", () => showSuggestions(input.value));
  input.addEventListener("focus", () => showSuggestions(input.value));
  input.addEventListener("keydown", (event) => {
    if (event.key === "Enter") run(input.value);
    if (event.key === "Escape") hideSuggestions();
  });
  $("go").addEventListener("click", () => run(input.value));
  $("surprise").addEventListener("click", pickRandom);
  document.addEventListener("click", (event) => {
    if (!$("suggestions").contains(event.target) && event.target !== input) hideSuggestions();
  });
}

async function boot() {
  bind();
  try {
    const response = await fetch("/api/topics");
    state.topics = (await response.json()).topics;
  } catch {
    flashError("Could not load topics. Start the server with uvicorn app:app.");
    return;
  }
  renderFilters();
  renderPlaceholder($("pro"), "Keywords the positive model associates with the topic.");
  renderPlaceholder($("con"), "Keywords the negative model associates with the topic.");
  $("topic").value = "AI regulation";
  run("AI regulation");
}

boot();
