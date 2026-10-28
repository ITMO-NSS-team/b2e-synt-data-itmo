const data = JSON.parse(document.getElementById("report-data").textContent);
const colors = {
  general_knowledge: "#98a2b3",
  skills_disabled: "#7f56d9",
  existing_skills: "#1570ef",
  generated_skills: "#f79009",
  existing_plus_generated: "#12b76a",
  existing_plus_catalog: "#039855",
};
const symbols = { abstract: "circle", medium: "square", detailed: "diamond", "—": "cross" };
const pct = (value, count, total) => value == null ? "—" :
  `${(value * 100).toFixed(1)}%${total ? ` (${count}/${total})` : ""}`;
const num = value => value == null ? "—" : Number(value).toFixed(1);
const esc = value => String(value).replace(/[&<>"']/g, char => ({
  "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
})[char]);
const shortMode = {
  general_knowledge: "general",
  skills_disabled: "disabled",
  existing_skills: "existing",
  generated_skills: "generated",
  existing_plus_generated: "existing + gen",
  existing_plus_catalog: "existing + catalog",
};
const configName = row => [row.mode, row.source, row.method, row.detail, row.variant]
  .filter(value => value && value !== "—").join(" / ");

document.getElementById("eval").textContent = data.eval_id;
document.getElementById("cards").innerHTML = [
  ["Всего runs", data.counts.runs],
  ["Оценено", data.counts.scored],
  ["Завершено", data.counts.completed],
  ["Требуют проверки", data.counts.review],
].map(([label, value]) => `<div class="card"><span class="muted">${label}</span><strong>${value}</strong></div>`).join("");

if (data.coverage) {
  const coverage = data.coverage;
  const section = document.getElementById("coverage");
  section.hidden = false;
  const catalogCoverage = (coverage.catalogs || []).map(row =>
    `${esc(row.source)}: ${row.matched}/${row.cases} кейсов с целевым generated skill`).join("; ");
  section.innerHTML = `<h2>Покрытие каталогами</h2><p>Кейсов выбрано: ${coverage.selected_cases} из ${coverage.source_cases}; без подходящего skill: ${coverage.excluded.length}; skills без подходящих кейсов: ${coverage.unmatched_artifacts.length}.${catalogCoverage ? ` ${catalogCoverage}.` : ""}</p>
    <details><summary>Непокрытые кейсы и skills</summary><ul>${coverage.excluded.map(row => `<li>${esc(row.case_id)} — ${esc(row.reason)}</li>`).join("")}${coverage.unmatched_artifacts.map(row => `<li>${esc(row.source)} / ${esc(row.skill_name)} (${esc(row.version)}) — нет подходящих кейсов среди выбранных</li>`).join("")}</ul></details>`;
}

document.getElementById("metrics").innerHTML = data.aggregates.map(row => `<tr>
  <td title="${esc(row.mode)}">${esc(shortMode[row.mode] || row.mode)}</td><td title="${esc(row.source)}">${esc(row.source)}</td><td title="${esc(row.method)}">${esc(row.method)}</td><td title="${esc(row.detail)}">${esc(row.detail)}</td><td>${esc(row.skill)}</td><td>${esc(row.variant)}</td><td>${row.runs}</td>
  <td>${pct(row.accuracy, row.accuracy_correct, row.accuracy_total)}</td>
  <td>${pct(row.format_valid, row.format_valid_count, row.format_valid_total)}</td>
  <td>${pct(row.llm_judge_accuracy, row.llm_judge_passed, row.llm_judge_scored)}</td>
  <td>${row.n_pairs || "—"}</td>
  <td>${row.delta_pp == null ? "—" : `${row.delta_pp > 0 ? "+" : ""}${row.delta_pp.toFixed(1)}`}</td>
  <td>${pct(row.any_skill_loaded, row.any_skill_loaded_count, row.any_skill_loaded_total)}</td><td>${pct(row.expected_skill_loaded, row.expected_skill_loaded_count, row.expected_skill_loaded_total)}</td>
  <td>${pct(row.generated_skill_loaded, row.generated_skill_loaded_count, row.generated_skill_loaded_total)}</td>
  <td>${row.median_latency_s == null ? "—" : `${num(row.median_latency_s)} с`}</td>
  <td>${num(row.median_mcp_queries)}</td><td>${num(row.median_iterations)}</td><td>${num(row.median_input_tokens)}</td><td>${num(row.median_output_tokens)}</td>
</tr>`).join("");

const plotConfig = {
  responsive: true,
  scrollZoom: true,
  displaylogo: false,
  toImageButtonOptions: { format: "png" },
};
const layout = (xTitle, yTitle, zTitle) => ({
  margin: { l: 68, r: 20, t: 24, b: 62 },
  paper_bgcolor: "transparent",
  plot_bgcolor: "#fff",
  hovermode: "closest",
  dragmode: "orbit",
  font: { family: "Inter, system-ui, sans-serif", color: "#344054" },
  scene: {
    xaxis: { title: { text: xTitle }, gridcolor: "#e7ebf1", zerolinecolor: "#98a2b3" },
    yaxis: { title: { text: yTitle }, gridcolor: "#e7ebf1", zerolinecolor: "#98a2b3" },
    zaxis: { title: { text: zTitle }, gridcolor: "#e7ebf1", zerolinecolor: "#98a2b3" },
  },
});
const legend = (rows, correctness = false) => {
  let html = Object.entries(colors)
    .filter(([mode]) => rows.some(row => row.mode === mode))
    .map(([mode, color]) => `<span><i class="dot" style="background:${color}"></i>${mode}</span>`)
    .join("");
  html += '<span><i class="shape">○</i>abstract</span><span><i class="shape">□</i>medium</span><span><i class="shape">◇</i>detailed</span>';
  if (correctness) {
    html += '<span><i class="dot" style="background:#fff;border:2px solid #067647"></i>правильно</span><span><i class="dot" style="background:#fff;border:2px solid #d92d20"></i>ошибка</span>';
  }
  return html;
};

const runs = data.runs.filter(row => row.accuracy != null && row.latency_s != null && row.iterations != null);
const runCustom = runs.map(row => [
  row.run_id, row.case_id, configName(row), row.accuracy, row.latency_s, row.mcp_queries, row.iterations,
]);
Plotly.newPlot("efficiency", [{
  type: "scatter",
  mode: "markers",
  x: runs.map(row => row.iterations),
  y: runs.map(row => row.latency_s),
  customdata: runCustom,
  marker: {
    size: 12,
    color: runs.map(row => colors[row.mode] || "#667085"),
    symbol: runs.map(row => symbols[row.detail] || "diamond"),
    line: { color: runs.map(row => row.accuracy === 1 ? "#067647" : "#d92d20"), width: 3 },
    opacity: 0.85,
  },
  hovertemplate: "<b>%{customdata[0]}</b><br>case: %{customdata[1]}<br>%{customdata[2]}<br>accuracy: %{customdata[3]}<br>time: %{customdata[4]:.1f} s<br>mcp_query: %{customdata[5]}<br>steps: %{customdata[6]}<extra></extra>",
}], {
  margin: { l: 68, r: 20, t: 24, b: 62 },
  paper_bgcolor: "transparent",
  plot_bgcolor: "#fff",
  hovermode: "closest",
  dragmode: "pan",
  font: { family: "Inter, system-ui, sans-serif", color: "#344054" },
  xaxis: { title: { text: "Шаги агента, шт." }, gridcolor: "#e7ebf1", zerolinecolor: "#98a2b3", automargin: true },
  yaxis: { title: { text: "Время выполнения, с" }, gridcolor: "#e7ebf1", zerolinecolor: "#98a2b3", automargin: true },
}, plotConfig);
document.getElementById("efficiency-legend").innerHTML = legend(data.runs, true);

const aggregates = data.aggregates.filter(row => row.accuracy != null && row.median_total_tokens != null && row.median_iterations != null);
const aggregateCustom = aggregates.map(row => [
  configName(row), row.accuracy, row.median_total_tokens, row.median_mcp_queries, row.median_iterations, row.runs,
]);
const qualityLayout = layout("Медиана total_tokens, токены", "Answer_accuracy, %", "Медиана шагов, шт.");
qualityLayout.scene.yaxis.range = [-0.03, 1.03];
qualityLayout.scene.yaxis.tickformat = ".0%";
Plotly.newPlot("quality-cost", [{
  type: "scatter3d",
  mode: "markers",
  x: aggregates.map(row => row.median_total_tokens),
  y: aggregates.map(row => row.accuracy),
  z: aggregates.map(row => row.median_iterations),
  customdata: aggregateCustom,
  marker: {
    size: 12,
    color: aggregates.map(row => colors[row.mode] || "#667085"),
    symbol: aggregates.map(row => symbols[row.detail] || "diamond"),
    line: { color: "#fff", width: 2 },
    opacity: 0.9,
  },
  hovertemplate: "<b>%{customdata[0]}</b><br>accuracy: %{customdata[1]:.1%}<br>tokens: %{customdata[2]:.1f}<br>median mcp_query: %{customdata[3]:.1f}<br>median steps: %{customdata[4]:.1f}<br>runs: %{customdata[5]}<extra></extra>",
}], qualityLayout, plotConfig);
document.getElementById("quality-legend").innerHTML = legend(aggregates);

document.getElementById("efficiency").on("plotly_click", event => {
  const row = event.points[0].customdata;
  document.getElementById("efficiency-selection").textContent = `${row[0]} | case: ${row[1]} | ${row[2]} | accuracy: ${row[3]} | time: ${num(row[4])} s | mcp_query: ${row[5]} | steps: ${row[6]}`;
});
document.getElementById("quality-cost").on("plotly_click", event => {
  const row = event.points[0].customdata;
  document.getElementById("quality-selection").textContent = `${row[0]} | accuracy: ${pct(row[1])} | tokens: ${num(row[2])} | median mcp_query: ${num(row[3])} | median steps: ${num(row[4])} | runs: ${row[5]}`;
});

document.querySelectorAll(".expand").forEach(button => button.addEventListener("click", () => {
  const panel = button.closest(".panel");
  const expanded = !panel.classList.contains("expanded");
  document.querySelectorAll(".panel.expanded").forEach(item => item.classList.remove("expanded"));
  document.querySelectorAll(".expand").forEach(item => { item.textContent = "Развернуть"; });
  panel.classList.toggle("expanded", expanded);
  document.body.classList.toggle("report-expanded", expanded);
  if (expanded) button.textContent = "Свернуть";
  requestAnimationFrame(() => Plotly.Plots.resize(panel.querySelector(".plot")));
}));
document.addEventListener("keydown", event => {
  if (event.key !== "Escape") return;
  document.querySelectorAll(".panel.expanded").forEach(item => item.classList.remove("expanded"));
  document.body.classList.remove("report-expanded");
  document.querySelectorAll(".expand").forEach(item => { item.textContent = "Развернуть"; });
  document.querySelectorAll(".plot").forEach(plot => Plotly.Plots.resize(plot));
});
