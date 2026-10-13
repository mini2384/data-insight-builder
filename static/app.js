/**
 * AI Data Insight Builder - Frontend Logic (Vanilla JavaScript)
 * Framework-free, Clean, Step-by-Step UI Controller
 */

// =====================================================================
// 애플리케이션 상태 (State)
// =====================================================================
const state = {
  selectedFile: null,
  diagnosis: null,
  suggestions: [],
  selectedTool: null,
  lastRunResult: null,
  currentChart: null,
  explainMode: "B",
  chatHistory: []
};

// =====================================================================
// 프론트엔드 콘솔 로거 (규격 엄격 준수)
// =====================================================================
function appLog(tag, message) {
  // 1. 브라우저 개발자 도구 Console 출력 (API Key, 개인정보 원문, CSV 전체 노출 금지)
  console.log(`[${tag}] ${message}`);

  // 2. 화면 하단 개발자 콘솔 패널 출력
  const consoleLogs = document.getElementById("consoleLogs");
  if (consoleLogs) {
    const entry = document.createElement("div");
    entry.className = "log-entry";

    const tagSpan = document.createElement("span");
    tagSpan.className = `log-tag ${tag.toLowerCase()}`;
    tagSpan.textContent = `[${tag}]`;

    const textSpan = document.createElement("span");
    textSpan.textContent = ` ${message}`;

    entry.appendChild(tagSpan);
    entry.appendChild(textSpan);
    consoleLogs.appendChild(entry);

    const consoleBody = document.getElementById("consoleBody");
    if (consoleBody) {
      consoleBody.scrollTop = consoleBody.scrollHeight;
    }
  }
}

function showAlert(message, type = "error") {
  const alertBox = document.getElementById("globalAlert");
  if (!alertBox) return;
  alertBox.className = `alert-banner ${type}`;
  alertBox.textContent = message;
  alertBox.classList.remove("hidden");
  setTimeout(() => {
    alertBox.classList.add("hidden");
  }, 6000);
}

// =====================================================================
// 초기화 및 헬스체크
// =====================================================================
document.addEventListener("DOMContentLoaded", async () => {
  appLog("PAGE", "페이지 로드 완료");
  initConsoleToggle();
  initDragAndDrop();
  initModeToggle();
  initFollowupChat();
  initCustomQuery();
  await checkServerHealth();
});

function initConsoleToggle() {
  const header = document.getElementById("consoleToggle");
  const body = document.getElementById("consoleBody");
  const icon = document.getElementById("consoleToggleIcon");
  if (header && body) {
    header.addEventListener("click", () => {
      const isHidden = body.classList.toggle("hidden");
      icon.textContent = isHidden ? "▲" : "▼";
    });
  }
}

async function checkServerHealth() {
  const serverBadge = document.getElementById("serverBadge");
  const apiKeyBadge = document.getElementById("apiKeyBadge");
  const modelBadge = document.getElementById("modelBadge");

  try {
    const res = await fetch("/health");
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();

    serverBadge.textContent = "● 서버 정상 가동";
    serverBadge.className = "status-badge online";

    if (data.has_api_key) {
      apiKeyBadge.textContent = "● API Key: 등록됨";
      apiKeyBadge.className = "status-badge online";
    } else {
      apiKeyBadge.textContent = "● API Key: 미설정 (.env 확인 필요)";
      apiKeyBadge.className = "status-badge offline";
    }

    modelBadge.textContent = `모델: ${data.model || "알 수 없음"}`;
    modelBadge.className = "status-badge model";
  } catch (err) {
    serverBadge.textContent = "● 서버 연결 실패";
    serverBadge.className = "status-badge offline";
    appLog("ERROR", `서버 헬스체크 실패: ${err.message}`);
  }
}

// =====================================================================
// 1단계: 파일 선택 & 드래그 앤 드롭 & 진단 (Inspect)
// =====================================================================
function initDragAndDrop() {
  const dropZone = document.getElementById("dropZone");
  const fileInput = document.getElementById("csvFileInput");
  const btnInspect = document.getElementById("btnInspect");
  const selectedFileBar = document.getElementById("selectedFileBar");
  const selectedFileName = document.getElementById("selectedFileName");

  if (!dropZone || !fileInput) return;

  dropZone.addEventListener("click", () => fileInput.click());

  dropZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropZone.classList.add("dragover");
  });

  dropZone.addEventListener("dragleave", () => {
    dropZone.classList.remove("dragover");
  });

  dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropZone.classList.remove("dragover");
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleFileSelected(e.dataTransfer.files[0]);
    }
  });

  fileInput.addEventListener("change", (e) => {
    if (e.target.files && e.target.files.length > 0) {
      handleFileSelected(e.target.files[0]);
    }
  });

  btnInspect.addEventListener("click", () => inspectFile());

  function handleFileSelected(file) {
    if (!file.name.toLowerCase().endsWith(".csv")) {
      appLog("ERROR", `잘못된 파일 선택: ${file.name} (CSV 아님)`);
      showAlert("CSV(.csv) 파일만 선택할 수 있습니다.");
      return;
    }

    state.selectedFile = file;
    const sizeStr = file.size >= 1024 * 1024 
      ? `${(file.size / (1024 * 1024)).toFixed(2)} MB` 
      : `${(file.size / 1024).toFixed(2)} KB`;

    appLog("UPLOAD", `파일 선택 (파일명: ${file.name}, 크기: ${sizeStr})`);
    selectedFileName.textContent = `📄 ${file.name} (${sizeStr})`;
    selectedFileBar.classList.remove("hidden");
  }
}

async function inspectFile() {
  if (!state.selectedFile) {
    showAlert("먼저 CSV 파일을 선택해 주세요.");
    return;
  }

  const inspectLoading = document.getElementById("inspectLoading");
  const diagnosisResults = document.getElementById("diagnosisResults");
  const btnInspect = document.getElementById("btnInspect");

  inspectLoading.classList.remove("hidden");
  diagnosisResults.classList.add("hidden");
  btnInspect.disabled = true;

  appLog("INSPECT", "진단 요청 시작 (POST /inspect)");

  const formData = new FormData();
  formData.append("file", state.selectedFile);

  try {
    const res = await fetch("/inspect", {
      method: "POST",
      body: formData
    });

    const resJson = await res.json();
    if (!res.ok) {
      throw new Error(resJson.error || "진단 도중 서버 오류가 발생했습니다.");
    }

    state.diagnosis = resJson.diagnosis;
    const diag = state.diagnosis;

    appLog("INSPECT", `진단 응답 수신 (행 수: ${diag.rows}, 열 수: ${diag.columns}, 경고 수: ${diag.warnings.length})`);
    renderDiagnosisResults(diag);

    diagnosisResults.classList.remove("hidden");
    document.getElementById("btnGoToSuggest").addEventListener("click", () => goToSuggestStep());
  } catch (err) {
    appLog("ERROR", `진단 실패: ${err.message}`);
    showAlert(`진단 오류: ${err.message}`);
  } finally {
    inspectLoading.classList.add("hidden");
    btnInspect.disabled = false;
  }
}

function renderDiagnosisResults(diag) {
  // 1. 경고 목록
  const warningsBox = document.getElementById("diagnosisWarningsBox");
  warningsBox.innerHTML = `<h4>⚠️ 발견된 주의사항 (${diag.warnings.length}건)</h4><ul>${diag.warnings.map(w => `<li>${w}</li>`).join("")}</ul>`;

  // 2. 통계 카드 그리드
  const statGrid = document.getElementById("diagnosisStatGrid");
  statGrid.innerHTML = `
    <div class="stat-card">
      <div class="stat-card-label">행 수 / 열 수</div>
      <div class="stat-card-value">${diag.rows.toLocaleString()}행 &bull; ${diag.columns}열</div>
    </div>
    <div class="stat-card">
      <div class="stat-card-label">성공 인코딩</div>
      <div class="stat-card-value">${diag.encoding}</div>
    </div>
    <div class="stat-card">
      <div class="stat-card-label">건너뛴 상단 줄</div>
      <div class="stat-card-value">${diag.skipped_rows}줄</div>
    </div>
    <div class="stat-card">
      <div class="stat-card-label">권장 기준연도</div>
      <div class="stat-card-value" style="color: var(--primary-color);">★ ${diag.recommended_year || "없음"}</div>
    </div>
    <div class="stat-card">
      <div class="stat-card-label">집계 그룹 행 수</div>
      <div class="stat-card-value" style="color: var(--warning-color);">${diag.aggregate_candidates.length}개</div>
    </div>
    <div class="stat-card">
      <div class="stat-card-label">중복 행 수</div>
      <div class="stat-card-value">${diag.duplicate_rows}개</div>
    </div>
  `;

  // 3. 미리보기 테이블 (상위 10행)
  const previewContainer = document.getElementById("previewTableContainer");
  if (diag.preview && diag.preview.length > 0) {
    const cols = Object.keys(diag.preview[0]);
    let tableHtml = "<table><thead><tr>";
    cols.forEach(c => tableHtml += `<th>${c}</th>`);
    tableHtml += "</tr></thead><tbody>";

    diag.preview.forEach(row => {
      tableHtml += "<tr>";
      cols.forEach(c => {
        const val = row[c];
        tableHtml += `<td>${val === null || val === undefined ? '<span style="color:#cbd5e1">-</span>' : val}</td>`;
      });
      tableHtml += "</tr>";
    });
    tableHtml += "</tbody></table>";
    previewContainer.innerHTML = tableHtml;
  }

  // 미리보기 접기/펼치기
  const btnTogglePreview = document.getElementById("btnTogglePreview");
  btnTogglePreview.onclick = () => {
    previewContainer.classList.toggle("hidden");
  };
}

// =====================================================================
// 2단계: AI 분석 제안 (Suggest)
// =====================================================================
async function goToSuggestStep() {
  document.getElementById("stepBadge1").classList.remove("active");
  document.getElementById("stepBadge2").classList.add("active");

  const sectionSuggest = document.getElementById("sectionSuggest");
  sectionSuggest.classList.remove("hidden");
  sectionSuggest.scrollIntoView({ behavior: "smooth" });

  await fetchSuggestions();
}

async function fetchSuggestions() {
  const suggestLoading = document.getElementById("suggestLoading");
  const suggestionsContainer = document.getElementById("suggestionsContainer");

  suggestLoading.classList.remove("hidden");
  suggestionsContainer.innerHTML = "";

  appLog("SUGGEST", "제안 요청 시작 (POST /suggest)");

  try {
    const res = await fetch("/suggest", { method: "POST" });
    const resJson = await res.json();

    if (!res.ok) {
      throw new Error(resJson.error || "분석 제안 생성 실패");
    }

    state.suggestions = resJson.suggestions || [];
    appLog("SUGGEST", `제안 응답 수신 (제안 개수: ${state.suggestions.length})`);

    renderSuggestionCards(state.suggestions);
    initManualCatalogForm();
  } catch (err) {
    appLog("ERROR", `제안 실패: ${err.message}`);
    showAlert(`제안 오류: ${err.message}`);
  } finally {
    suggestLoading.classList.add("hidden");
  }
}

function renderSuggestionCards(suggestions) {
  const container = document.getElementById("suggestionsContainer");
  container.innerHTML = "";

  if (suggestions.length === 0) {
    container.innerHTML = "<p>제안된 분석 항목이 없습니다.</p>";
    return;
  }

  suggestions.forEach((item, idx) => {
    const card = document.createElement("div");
    card.className = "suggestion-card";
    card.dataset.index = idx;

    const refText = item.reference_period ? `기준: ${item.reference_period}년` : "전체 시점";

    card.innerHTML = `
      <div class="card-tool-name">${idx + 1}. ${item.name || item.tool}</div>
      <div class="card-ref-badge">${refText}</div>
      <p class="card-why"><strong>💡 제안 이유:</strong> ${item.why}</p>
      <div class="card-caution"><strong>⚠️ 주의사항:</strong> ${item.caution}</div>
    `;

    card.addEventListener("click", () => {
      document.querySelectorAll(".suggestion-card").forEach(c => c.classList.remove("selected"));
      card.classList.add("selected");
      selectToolForRun(item);
    });

    container.appendChild(card);
  });

  // 첫 번째 제안 자동 선택
  if (suggestions.length > 0) {
    container.children[0].click();
  }
}

function selectToolForRun(toolItem) {
  state.selectedTool = toolItem;
  const panel = document.getElementById("selectedToolPanel");
  const title = document.getElementById("currentToolTitle");
  const period = document.getElementById("currentToolPeriod");

  title.textContent = toolItem.name || toolItem.tool;
  period.textContent = toolItem.reference_period ? `${toolItem.reference_period}년 기준` : "전체 시점";
  panel.classList.remove("hidden");

  const btnRunTool = document.getElementById("btnRunTool");
  btnRunTool.onclick = () => runSelectedTool();
}

function initManualCatalogForm() {
  const btnToggle = document.getElementById("btnToggleManualCatalog");
  const form = document.getElementById("manualCatalogForm");
  const select = document.getElementById("manualToolSelect");
  const btnApply = document.getElementById("btnApplyManual");

  btnToggle.onclick = () => form.classList.toggle("hidden");

  // 12개 카탈로그 옵션 구성
  const catalog = [
    { id: "top_bottom_n", name: "기준연도 상위·하위 N 순위" },
    { id: "trend_line", name: "시계열 추이 비교 (선 차트)" },
    { id: "group_summary", name: "그룹별 평균·합계 비교" },
    { id: "change_rate", name: "두 시점 간 변화량·성장률" },
    { id: "distribution", name: "구간별 빈도 분포 (히스토그램)" },
    { id: "correlation_scatter", name: "상관계수 및 산점도 분석" },
    { id: "compare_one_vs_all", name: "특정 항목 vs 전체 비교" },
    { id: "describe_numeric", name: "기초 통계량 계산" },
    { id: "year_coverage", name: "연도별 데이터 충실도 분석" },
    { id: "detect_aggregates", name: "집계 행 후보 탐지" },
    { id: "reshape_wide_to_long", name: "연도 열을 행으로 접기" },
    { id: "quality_report", name: "데이터 품질 종합 리포트" }
  ];

  select.innerHTML = catalog.map(c => `<option value="${c.id}">${c.name}</option>`).join("");

  btnApply.onclick = () => {
    const toolId = select.value;
    const refPeriod = document.getElementById("manualRefPeriod").value.trim() || (state.diagnosis ? state.diagnosis.recommended_year : "2024");
    const excludeAgg = document.getElementById("manualExcludeAgg").checked;

    const manualItem = {
      tool: toolId,
      name: catalog.find(c => c.id === toolId)?.name,
      params: {
        reference_period: refPeriod,
        exclude_aggregates: excludeAgg
      },
      reference_period: refPeriod,
      why: "사용자가 수동 카탈로그에서 직접 선택한 분석입니다.",
      caution: "기준 시점에 데이터가 없는 국가는 분석에서 제외됩니다."
    };

    document.querySelectorAll(".suggestion-card").forEach(c => c.classList.remove("selected"));
    selectToolForRun(manualItem);
    form.classList.add("hidden");
  };
}

// =====================================================================
// 3단계: 도구 실행 (Run & Chart.js)
// =====================================================================
async function runSelectedTool() {
  if (!state.selectedTool) {
    showAlert("실행할 분석 도구를 선택해 주세요.");
    return;
  }

  const tool = state.selectedTool;
  const sectionRun = document.getElementById("sectionRun");
  const runLoading = document.getElementById("runLoading");
  const runResultsBox = document.getElementById("runResultsBox");

  document.getElementById("stepBadge2").classList.remove("active");
  document.getElementById("stepBadge3").classList.add("active");

  sectionRun.classList.remove("hidden");
  sectionRun.scrollIntoView({ behavior: "smooth" });

  runLoading.classList.remove("hidden");
  runResultsBox.classList.add("hidden");

  appLog("RUN", `도구 실행 시작 (도구명: ${tool.tool}, 기준시점: ${tool.reference_period || "없음"})`);

  try {
    const res = await fetch("/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        tool: tool.tool,
        params: tool.params || {}
      })
    });

    const resJson = await res.json();
    if (!res.ok) {
      throw new Error(resJson.error || "분석 실행 실패");
    }

    state.lastRunResult = resJson.data;
    const data = state.lastRunResult;

    appLog("RUN", `도구 결과 수신 (사용 행 수: ${data.used_rows}, 제외 행 수: ${data.excluded_rows})`);

    // 1. 감사 카드 렌더링
    renderAuditCard(data);

    // 2. Chart.js 렌더링
    renderChart(data.result?.chart);

    // 3. 세부 계산 표 렌더링
    renderCalcTable(data.result);

    runResultsBox.classList.remove("hidden");
    document.getElementById("btnGoToExplain").onclick = () => goToExplainStep();
  } catch (err) {
    appLog("ERROR", `도구 실행 실패: ${err.message}`);
    showAlert(`실행 오류: ${err.message}`);
  } finally {
    runLoading.classList.add("hidden");
  }
}

function renderAuditCard(data) {
  document.getElementById("auditRefPeriod").textContent = data.reference_period || "전체";
  document.getElementById("auditUsedRows").textContent = `${data.used_rows}개 행`;
  document.getElementById("auditExcludedRows").textContent = `${data.excluded_rows}개 행 제외됨`;
  document.getElementById("auditReason").textContent = data.exclusion_reason || "없음";
  document.getElementById("auditCaution").textContent = data.caution || "없음";
}

function renderChart(chartConfig) {
  const chartWrapper = document.getElementById("chartWrapper");
  const canvas = document.getElementById("mainChart");

  if (!chartConfig || !chartConfig.data && !chartConfig.datasets) {
    chartWrapper.classList.add("hidden");
    return;
  }

  chartWrapper.classList.remove("hidden");

  // 이전 차트 인스턴스 파괴
  if (state.currentChart) {
    state.currentChart.destroy();
    state.currentChart = null;
  }

  const ctx = canvas.getContext("2d");

  // Chart.js 옵션 표준화
  const options = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        display: chartConfig.type !== "pie",
        position: "top"
      },
      tooltip: {
        enabled: true
      }
    }
  };

  if (chartConfig.indexAxis === "y") {
    options.indexAxis = "y";
  }

  const config = {
    type: chartConfig.type || "bar",
    data: {
      labels: chartConfig.labels || [],
      datasets: chartConfig.datasets || []
    },
    options: options
  };

  state.currentChart = new Chart(ctx, config);
  appLog("CHART", `차트 렌더 완료 (차트 종류: ${config.type})`);
}

function renderCalcTable(result) {
  const container = document.getElementById("calcTableContainer");
  if (!result) {
    container.innerHTML = "";
    return;
  }

  let tableData = [];
  if (result.table_rows) {
    tableData = result.table_rows;
  } else if (result.top_items && result.bottom_items) {
    tableData = result.top_items.concat(result.bottom_items);
  } else if (result.group_details) {
    tableData = result.group_details;
  } else if (result.top_growth_items) {
    tableData = result.top_growth_items;
  } else if (result.distribution_table) {
    tableData = result.distribution_table;
  } else if (result.preview) {
    tableData = result.preview;
  } else if (result.coverage_table) {
    tableData = result.coverage_table.slice(-10);
  }

  if (tableData.length === 0) {
    container.innerHTML = `<pre style="padding:10px; font-size:0.8rem;">${JSON.stringify(result, null, 2)}</pre>`;
    return;
  }

  const cols = Object.keys(tableData[0]).filter(c => c !== "is_target");
  let html = "<table><thead><tr>";
  cols.forEach(c => html += `<th>${c}</th>`);
  html += "</tr></thead><tbody>";

  tableData.forEach(row => {
    const isTarget = row.is_target;
    html += `<tr class="${isTarget ? 'table-target-row' : ''}">`;
    cols.forEach(c => {
      const val = row[c];
      html += `<td>${val === null || val === undefined ? "-" : val}</td>`;
    });
    html += "</tr>";
  });
  html += "</tbody></table>";
  container.innerHTML = html;
}

// =====================================================================
// 4단계: AI 결과 해석 (Explain: Mode A vs Mode B)
// =====================================================================
function initModeToggle() {
  const btnA = document.getElementById("btnModeA");
  const btnB = document.getElementById("btnModeB");
  const explainer = document.getElementById("modeExplainer");

  function updateModeUI() {
    if (state.explainMode === "B") {
      btnB.classList.add("active");
      btnA.classList.remove("active");
      explainer.innerHTML = `
        <strong>🛡️ 모드 B (정식 팩트 기반 보고서):</strong><br>
        Pandas가 계산한 결과(수치, 제외된 72개 행, ITU 공식 출처)만을 Gemini에게 전달합니다. 
        AI가 가짜 숫자를 지어내지 못하도록 <strong>엄격한 8대 규칙과 7대 고정 헤더(##)</strong>로 해석합니다.
      `;
    } else {
      btnA.classList.add("active");
      btnB.classList.remove("active");
      explainer.innerHTML = `
        <strong>⚠️ 모드 A (비교용 자유 추정 모드):</strong><br>
        Pandas 계산 결과를 주지 않고 진단 요약만 준 뒤 "보고서를 써달라"고 요청합니다. 
        AI가 사전 지식과 추정으로 서술하므로 모드 B와 비교하여 <strong>환각(Hallucination) 위험성</strong>을 체감할 수 있습니다.
      `;
    }
  }

  btnA.onclick = () => { state.explainMode = "A"; updateModeUI(); };
  btnB.onclick = () => { state.explainMode = "B"; updateModeUI(); };
  updateModeUI();

  document.getElementById("btnGenerateExplain").onclick = () => generateExplanation();
}

function goToExplainStep() {
  document.getElementById("stepBadge3").classList.remove("active");
  document.getElementById("stepBadge4").classList.add("active");

  const sectionExplain = document.getElementById("sectionExplain");
  sectionExplain.classList.remove("hidden");
  sectionExplain.scrollIntoView({ behavior: "smooth" });
}

async function generateExplanation() {
  const explainLoading = document.getElementById("explainLoading");
  const explainLoadingText = document.getElementById("explainLoadingText");
  const reportContainer = document.getElementById("reportContainer");
  const reportContent = document.getElementById("reportContent");
  const reportBadgeMode = document.getElementById("reportBadgeMode");

  explainLoading.classList.remove("hidden");
  reportContainer.classList.add("hidden");

  explainLoadingText.textContent = state.explainMode === "B"
    ? "Gemini가 계산 결과만을 바탕으로 엄격한 7대 항목 보고서를 작성하고 있습니다..."
    : "Gemini가 일반 지식으로 자유 보고서를 작성하고 있습니다...";

  appLog("EXPLAIN", `해석 요청 시작 (모드: ${state.explainMode})`);

  try {
    const res = await fetch("/explain", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        mode: state.explainMode,
        calculation_result: state.lastRunResult
      })
    });

    const resJson = await res.json();
    if (!res.ok) {
      throw new Error(resJson.error || "해석 보고서 생성 실패");
    }

    appLog("EXPLAIN", `해석 응답 수신 (모드: ${resJson.mode})`);

    const markdownText = resJson.explanation || "";
    // Marked 라이브러리로 HTML 파싱
    reportContent.innerHTML = window.marked ? window.marked.parse(markdownText) : `<pre>${markdownText}</pre>`;
    reportBadgeMode.textContent = resJson.mode === "B" ? "모드 B (팩트 검증 보고서)" : "모드 A (자유 추정 보고서)";

    reportContainer.classList.remove("hidden");

    // 후속 질문 대화창 표시 (앱의 맨 아랫부분)
    const followupChat = document.getElementById("sectionFollowupChat");
    if (followupChat) {
      followupChat.classList.remove("hidden");
      const badge = document.getElementById("chatReportContextBadge");
      if (badge) {
        badge.textContent = resJson.mode === "B" ? "모드 B 보고서 연동" : "모드 A 보고서 연동";
      }
      document.getElementById("stepBadge4")?.classList.remove("active");
      document.getElementById("stepBadge5")?.classList.add("active");
    }

    // 클립보드 복사 버튼
    document.getElementById("btnCopyReport").onclick = () => {
      navigator.clipboard.writeText(markdownText).then(() => {
        alert("보고서 전문이 클립보드에 복사되었습니다!");
      });
    };
  } catch (err) {
    appLog("ERROR", `해석 실패: ${err.message}`);
    showAlert(`해석 오류: ${err.message}`);
  } finally {
    explainLoading.classList.add("hidden");
  }
}

// =====================================================================
// 5단계: AI 데이터 튜터 후속 대화 (Chat Q&A - 중학생 눈높이 해설)
// =====================================================================
function initFollowupChat() {
  const form = document.getElementById("chatInputForm");
  const input = document.getElementById("chatInputText");
  const chips = document.querySelectorAll(".chip-btn");

  if (form) {
    form.addEventListener("submit", (e) => {
      e.preventDefault();
      const text = input.value.trim();
      if (!text) return;
      input.value = "";
      sendChatMessage(text);
    });
  }

  chips.forEach(chip => {
    chip.addEventListener("click", () => {
      const q = chip.getAttribute("data-q");
      if (q) {
        sendChatMessage(q);
      }
    });
  });
}

async function sendChatMessage(message) {
  const chatMessages = document.getElementById("chatMessages");
  const btnSend = document.getElementById("btnSendChat");
  const input = document.getElementById("chatInputText");

  // 1. 사용자 메시지 말풍선 추가
  const userMsgEl = document.createElement("div");
  userMsgEl.className = "chat-msg user";
  userMsgEl.innerHTML = `
    <div class="msg-avatar">🧑‍🎓</div>
    <div class="msg-bubble">${escapeHtml(message)}</div>
  `;
  chatMessages.appendChild(userMsgEl);

  // 2. AI 생각 중 로딩 말풍선 추가
  const aiMsgEl = document.createElement("div");
  aiMsgEl.className = "chat-msg ai";
  aiMsgEl.innerHTML = `
    <div class="msg-avatar">🤖</div>
    <div class="msg-bubble loading-msg">
      <em>선생님이 중학생 눈높이에 맞춰 알기 쉽게 설명할 내용을 정리하고 있어요... 💭</em>
    </div>
  `;
  chatMessages.appendChild(aiMsgEl);
  chatMessages.scrollTop = chatMessages.scrollHeight;

  // 비활성화
  if (btnSend) btnSend.disabled = true;
  if (input) input.disabled = true;

  appLog("CHAT", `후속 질문 전송: "${message.slice(0, 30)}..." (모드: ${state.explainMode})`);

  try {
    const res = await fetch("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message: message,
        chat_history: state.chatHistory,
        mode: state.explainMode
      })
    });

    const resJson = await res.json();
    if (!res.ok) {
      throw new Error(resJson.error || "답변을 가져오지 못했습니다.");
    }

    const replyText = resJson.reply || "";
    appLog("CHAT", `AI 답변 수신 완료 (${replyText.length}자)`);

    const bubbleEl = aiMsgEl.querySelector(".msg-bubble");
    bubbleEl.classList.remove("loading-msg");
    bubbleEl.innerHTML = window.marked ? window.marked.parse(replyText) : replyText.replace(/\n/g, "<br>");

    state.chatHistory.push({ role: "user", content: message });
    state.chatHistory.push({ role: "model", content: replyText });

  } catch (err) {
    appLog("ERROR", `채팅 실패: ${err.message}`);
    const bubbleEl = aiMsgEl.querySelector(".msg-bubble");
    bubbleEl.classList.remove("loading-msg");
    bubbleEl.innerHTML = `<span style="color: var(--error-color);">⚠️ 죄송해요, 답변을 생성하는 도중 오류가 발생했어요: ${err.message}</span>`;
  } finally {
    if (btnSend) btnSend.disabled = false;
    if (input) {
      input.disabled = false;
      input.focus();
    }
    chatMessages.scrollTop = chatMessages.scrollHeight;
  }
}

// =====================================================================
// 6단계: 맞춤 자연어 데이터 질의응답 (Custom Data Query)
// =====================================================================
function initCustomQuery() {
  const form = document.getElementById("customQueryForm");
  const input = document.getElementById("customQueryInput");
  const chips = document.querySelectorAll(".query-chip");

  if (form) {
    form.addEventListener("submit", (e) => {
      e.preventDefault();
      const question = input.value.trim();
      if (!question) {
        showAlert("조회하고 싶은 질문을 입력해 주세요.");
        return;
      }
      executeCustomQuery(question);
    });
  }

  chips.forEach(chip => {
    chip.addEventListener("click", () => {
      const q = chip.getAttribute("data-query");
      if (q) {
        if (input) input.value = q;
        executeCustomQuery(q);
      }
    });
  });
}

async function executeCustomQuery(question) {
  const loading = document.getElementById("customQueryLoading");
  const resultBox = document.getElementById("customQueryResult");
  const yearBadge = document.getElementById("customQueryYearBadge");
  const tableWrapper = document.getElementById("customQueryTableWrapper");
  const explainContent = document.getElementById("customQueryExplainContent");
  const btnSubmit = document.getElementById("btnRunCustomQuery");

  loading.classList.remove("hidden");
  resultBox.classList.add("hidden");
  if (btnSubmit) btnSubmit.disabled = true;

  appLog("QUERY", `맞춤 질의 요청 시작: "${question}"`);

  try {
    const res = await fetch("/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: question })
    });

    const resJson = await res.json();
    if (!res.ok) {
      throw new Error(resJson.error || "맞춤 질의 처리에 실패했습니다.");
    }

    appLog("QUERY", `맞춤 질의 응답 수신 (기준연도: ${resJson.year}, 행 수: ${resJson.table_rows?.length})`);

    // 1. 📈 3단계(Section 3: 분석 결과 & 차트 시각화)에 즉시 반영!
    if (resJson.run_result) {
      state.lastRunResult = resJson.run_result;

      const sectionRun = document.getElementById("sectionRun");
      const runResultsBox = document.getElementById("runResultsBox");
      const stepBadge2 = document.getElementById("stepBadge2");
      const stepBadge3 = document.getElementById("stepBadge3");

      if (stepBadge2) stepBadge2.classList.remove("active");
      if (stepBadge3) stepBadge3.classList.add("active");

      if (sectionRun) {
        sectionRun.classList.remove("hidden");
      }

      // 3단계 감사 카드 렌더링
      renderAuditCard(resJson.run_result);

      // 3단계 Chart.js 차트 렌더링 (가로 막대 차트: 상위 10개국 + 한국 13위 강조!)
      renderChart(resJson.run_result.result?.chart);

      // 3단계 세부 계산 데이터 표 렌더링
      renderCalcTable(resJson.run_result.result);

      if (runResultsBox) {
        runResultsBox.classList.remove("hidden");
      }

      // 4단계로 이동하는 버튼 이벤트 연결
      const btnGoToExplain = document.getElementById("btnGoToExplain");
      if (btnGoToExplain) {
        btnGoToExplain.onclick = () => goToExplainStep();
      }
    }

    // 2. 질문 결과 카드에도 연도 배지, 표 및 AI 선생님 해설 렌더링
    if (yearBadge) {
      yearBadge.textContent = `${resJson.year}년 기준 팩트`;
    }
    renderCustomQueryTable(resJson.table_rows, tableWrapper);

    const expMarkdown = resJson.explanation || "";
    explainContent.innerHTML = window.marked ? window.marked.parse(expMarkdown) : `<pre>${expMarkdown}</pre>`;
    resultBox.classList.remove("hidden");

    // 3. 사용자가 차트와 계산 결과를 바로 볼 수 있도록 3단계 시각화 영역으로 부드럽게 스크롤!
    const sectionRun = document.getElementById("sectionRun");
    if (sectionRun) {
      sectionRun.scrollIntoView({ behavior: "smooth" });
    }

  } catch (err) {
    appLog("ERROR", `맞춤 질의 실패: ${err.message}`);
    showAlert(`맞춤 질의 오류: ${err.message}`);
  } finally {
    loading.classList.add("hidden");
    if (btnSubmit) btnSubmit.disabled = false;
  }
}

function renderCustomQueryTable(rows, container) {
  if (!rows || rows.length === 0) {
    container.innerHTML = "<p>표시할 데이터가 없습니다.</p>";
    return;
  }

  let html = `
    <table class="data-table">
      <thead>
        <tr>
          <th style="width: 80px;">순위</th>
          <th>국가명</th>
          <th style="width: 140px;">인터넷 이용률</th>
          <th>비고</th>
        </tr>
      </thead>
      <tbody>
  `;

  rows.forEach(r => {
    const rowClass = r.is_target ? "table-target-row" : "";
    html += `
      <tr class="${rowClass}">
        <td><strong>${r.rank}</strong></td>
        <td>${escapeHtml(r.country)}</td>
        <td><strong>${r.value}</strong></td>
        <td>${escapeHtml(r.note || "")}</td>
      </tr>
    `;
  });

  html += "</tbody></table>";
  container.innerHTML = html;
}

function escapeHtml(str) {
  return String(str || "").replace(/[&<>'"]/g, 
    tag => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#39;', '"': '&quot;' }[tag] || tag)
  );
}
