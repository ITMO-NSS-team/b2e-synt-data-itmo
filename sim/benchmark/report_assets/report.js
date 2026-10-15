const data = JSON.parse(document.getElementById("report-data").textContent);
const colors = {
  general_knowledge: "#98a2b3",
  skills_disabled: "#7f56d9",
  existing_skills: "#1570ef",
  generated_skills: "#f79009",
  existing_plus_generated: "#12b76a",
};
const symbols = { abstract: "circle", medium: "square", detailed: "triangle-up", "—": "diamond" };
const pct = value => value == null ? "—" : `${(value * 100).toFixed(1)}%`;
const num = value => value == null ? "—" : Number(value).toFixed(1);
const esc = value => String(value).replace(/[&<>"']/g, char => ({
  "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
})[char]);
const configName = row => [row.mode, row.method, row.detail]
  .filter(value => value && value !== "—").join(" / ");

document.getElementById("eval").textContent = data.eval_id;
document.getElementById("cards").innerHTML = [
  ["Всего runs", data.counts.runs],
  ["Оценено", data.counts.scored],
  ["Завершено", data.counts.completed],
  ["Требуют проверки", data.counts.review],
].map(([label, value]) => `<div class="card"><span class="muted">${label}</span><strong>${value}</strong></div>`).join("");

document.getElementById("metrics").innerHTML = data.aggregates.map(row => `<tr>
  <td>${esc(row.mode)}</td><td>${esc(row.method)}</td><td>${esc(row.detail)}</td><td>${row.runs}</td>
  <td>${pct(row.accuracy)}</td>
  <td>${row.delta_pp == null ? "—" : `${row.delta_pp > 0 ? "+" : ""}${row.delta_pp.toFixed(1)}`}</td>
  <td>${pct(row.any_skill_loaded)}</td><td>${pct(row.expected_skill_loaded)}</td>
  <td>${row.median_latency_s == null ? "—" : `${num(row.median_latency_s)} с`}</td>
  <td>${num(row.median_mcp_queries)}</td><td>${num(row.median_total_tokens)}</td>
</tr>`).join("");

const plotConfig = {
  responsive: true,
  scrollZoom: true,
  displaylogo: false,
  toImageButtonOptions: { format: "png" },
};
const layout = (xTitle, yTitle) => ({
  margin: { l: 68, r: 20, t: 24, b: 62 },
  paper_bgcolor: "transparent",
  plot_bgcolor: "#fff",
  hovermode: "closest",
  dragmode: "pan",
  font: { family: "Inter, system-ui, sans-serif", color: "#344054" },
  xaxis: { title: { text: xTitle }, gridcolor: "#e7ebf1", zerolinecolor: "#98a2b3", automargin: true },
  yaxis: { title: { text: yTitle }, gridcolor: "#e7ebf1", zerolinecolor: "#98a2b3", automargin: true },
});
const legend = (rows, correctness = false) => {
  let html = Object.entries(colors)
    .filter(([mode]) => rows.some(row => row.mode === mode))
    .map(([mode, color]) => `<span><i class="dot" style="background:${color}"></i>${mode}</span>`)
    .join("");
  html += '<span><i class="shape">○</i>abstract</span><span><i class="shape">□</i>medium</span><span><i class="shape">△</i>detailed</span>';
  if (correctness) {
    html += '<span><i class="dot" style="background:#fff;border:2px solid #067647"></i>правильно</span><span><i class="dot" style="background:#fff;border:2px solid #d92d20"></i>ошибка</span>';
  }
  return html;
};

const runs = data.runs.filter(row => row.accuracy != null && row.latency_s != null && row.mcp_queries != null);
const runCustom = runs.map(row => [
  row.run_id, row.case_id, configName(row), row.accuracy, row.latency_s, row.mcp_queries,
]);
Plotly.newPlot("efficiency", [{
  type: "scatter",
  mode: "markers",
  x: runs.map(row => row.mcp_queries),
  y: runs.map(row => row.latency_s),
  customdata: runCustom,
  marker: {
    size: 12,
    color: runs.map(row => colors[row.mode] || "#667085"),
    symbol: runs.map(row => symbols[row.detail] || "diamond"),
    line: { color: runs.map(row => row.accuracy === 1 ? "#067647" : "#d92d20"), width: 3 },
    opacity: 0.85,
  },
  hovertemplate: "<b>%{customdata[0]}</b><br>case: %{customdata[1]}<br>%{customdata[2]}<br>accuracy: %{customdata[3]}<br>time: %{customdata[4]:.1f} s<br>mcp_query: %{customdata[5]}<extra></extra>",
}], layout("Число вызовов mcp_query, шт.", "Время выполнения, с"), plotConfig);
document.getElementById("efficiency-legend").innerHTML = legend(data.runs, true);

const aggregates = data.aggregates.filter(row => row.accuracy != null && row.median_total_tokens != null);
const aggregateCustom = aggregates.map(row => [
  configName(row), row.accuracy, row.median_total_tokens, row.median_mcp_queries, row.runs,
]);
const qualityLayout = layout("Медиана total_tokens, токены", "Answer_accuracy, %");
qualityLayout.yaxis.range = [-0.03, 1.03];
qualityLayout.yaxis.tickformat = ".0%";
Plotly.newPlot("quality-cost", [{
  type: "scatter",
  mode: "markers",
  x: aggregates.map(row => row.median_total_tokens),
  y: aggregates.map(row => row.accuracy),
  customdata: aggregateCustom,
  marker: {
    size: 12,
    color: aggregates.map(row => colors[row.mode] || "#667085"),
    symbol: aggregates.map(row => symbols[row.detail] || "diamond"),
    line: { color: "#fff", width: 2 },
    opacity: 0.9,
  },
  hovertemplate: "<b>%{customdata[0]}</b><br>accuracy: %{customdata[1]:.1%}<br>tokens: %{customdata[2]:.1f}<br>median mcp_query: %{customdata[3]:.1f}<br>runs: %{customdata[4]}<extra></extra>",
}], qualityLayout, plotConfig);
document.getElementById("quality-legend").innerHTML = legend(aggregates);

document.getElementById("efficiency").on("plotly_click", event => {
  const row = event.points[0].customdata;
  document.getElementById("efficiency-selection").textContent = `${row[0]} | case: ${row[1]} | ${row[2]} | accuracy: ${row[3]} | time: ${num(row[4])} s | mcp_query: ${row[5]}`;
});
document.getElementById("quality-cost").on("plotly_click", event => {
  const row = event.points[0].customdata;
  document.getElementById("quality-selection").textContent = `${row[0]} | accuracy: ${pct(row[1])} | tokens: ${num(row[2])} | median mcp_query: ${num(row[3])} | runs: ${row[4]}`;
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
