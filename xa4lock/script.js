// Ссылка на файл мода (собирается mod/build.py). Пусто — кнопка показывает «скоро».
const DOWNLOAD_URL = "downloads/Xa4lock-0.5.zip";

const grid = document.getElementById("grid");
const filters = document.getElementById("filters");
const toast = document.getElementById("toast");

// Цвета флага Армении по кругу для аватаров
const COLORS = ["var(--red)", "var(--blue)", "var(--orange)"];

function card(h, i) {
  const el = document.createElement("article");
  el.className = "card";
  el.dataset.role = h.role;
  el.innerHTML = `
    <div class="avatar" style="--c:${COLORS[i % 3]}"><span>${h.glyph}</span></div>
    <div class="card-body">
      <p class="orig">вместо ${h.orig}</p>
      <h3>${h.name}</h3>
      <span class="role">${h.role}</span>
      ${h.soon ? '<span class="role soon">скоро в игре</span>' : ""}
      <p>${h.bio}</p>
      ${h.abilities.length ? `<ol class="abilities">${h.abilities.map(([ru, orig], k) =>
        `<li${k === 3 ? ' class="ult"' : ""} title="${orig}">${ru}</li>`).join("")}</ol>` :
        '<p class="note">Умения появятся после выхода героя.</p>'}
    </div>`;
  return el;
}

HEROES.forEach((h, i) => grid.appendChild(card(h, i)));
document.getElementById("stat-heroes").textContent = HEROES.length;

const roles = ["Все", ...new Set(HEROES.map(h => h.role))];
roles.forEach((r, i) => {
  const b = document.createElement("button");
  b.textContent = r;
  b.setAttribute("aria-pressed", i === 0);
  b.addEventListener("click", () => {
    filters.querySelectorAll("button").forEach(x => x.setAttribute("aria-pressed", x === b));
    grid.querySelectorAll(".card").forEach(c => {
      c.hidden = r !== "Все" && c.dataset.role !== r;
    });
  });
  filters.appendChild(b);
});

function showToast(text) {
  toast.textContent = text;
  toast.classList.add("show");
  clearTimeout(showToast.t);
  showToast.t = setTimeout(() => toast.classList.remove("show"), 2800);
}

const dl = document.getElementById("download-btn");
if (DOWNLOAD_URL) {
  dl.href = DOWNLOAD_URL;
} else {
  document.getElementById("download-note").textContent = "Первый релиз скоро. Следите за обновлениями.";
  dl.addEventListener("click", e => {
    e.preventDefault();
    showToast("Релиз скоро — мы ещё дожариваем шашлык 🍢");
  });
}
