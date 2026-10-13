import os
import io
import csv
import re
import json
import logging
import traceback
import inspect
import glob
from flask import Flask, jsonify, request, render_template
import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.genai import types
from tools import TOOL_CATALOG, TOOL_REGISTRY, sanitize_for_json, filter_aggregates_if_needed

# .env 환경변수 로드
load_dotenv()

# 로깅 설정 (print() 대신 logging 사용)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# 파일 업로드 용량 제한 설정 (기본 10MB)
max_mb = int(os.getenv("MAX_UPLOAD_MB", 10))
app.config["MAX_CONTENT_LENGTH"] = max_mb * 1024 * 1024

# 메모리 데이터셋 보관소 (디스크에 영구 저장하지 않고 서버 메모리에 1개 데이터셋만 보관)
CURRENT_DATASET = {
    "df": None,
    "filename": None,
    "encoding": None,
    "skipped_rows": 0,
    "diagnosis": None,
    "last_run_result": None,
    "last_explanation": "",
    "last_explain_mode": "B"
}

# 시도할 인코딩 우선순위 목록
SUPPORTED_ENCODINGS = ["utf-8-sig", "utf-8", "cp949", "euc-kr"]

# 집계 행(대륙, 소득 그룹, 연합체 등) 탐지 키워드 목록
AGG_KEYWORDS = [
    "world", "oecd members", "euro area", "european union",
    "high income", "low income", "middle income", "income",
    "area", "total", "dividend", "small states",
    "ida & ibrd", "ibrd only", "ida only", "ida total", "not classified"
]


def try_detect_encoding(file_bytes):
    """지정된 인코딩을 순서대로 시도하여 성공한 텍스트와 인코딩명을 반환"""
    for enc in SUPPORTED_ENCODINGS:
        try:
            text = file_bytes.decode(enc)
            logger.info(f"[ENCODING] {enc} 성공")
            return text, enc
        except UnicodeDecodeError:
            logger.info(f"[ENCODING] {enc} 실패")
    return None, None


def detect_header_row(text):
    """
    상단 여러 줄을 검사하여 표의 실제 제목(헤더)으로 보이는 줄의 인덱스를 반환.
    건너뛴 줄 수(skiprows)와 감지된 헤더 컬럼 목록을 반환.
    """
    reader = csv.reader(io.StringIO(text))
    lines = []
    for _ in range(30):
        try:
            lines.append(next(reader))
        except StopIteration:
            break

    if not lines:
        return 0, []

    col_counts = [len([col for col in row if col.strip()]) for row in lines]
    max_cols = max(col_counts) if col_counts else 0

    if max_cols <= 1:
        return 0, lines[0] if lines else []

    header_idx = 0
    for idx, count in enumerate(col_counts):
        if count >= max_cols * 0.8 and count > 1:
            header_idx = idx
            break

    logger.info(f"[HEADER] 헤더 줄 감지 완료: {header_idx + 1}번째 줄 (건너뛴 줄 수: {header_idx})")
    return header_idx, lines[header_idx]


@app.route("/", methods=["GET"])
def index():
    logger.info("[REQUEST] GET /")
    logger.info("[RESPONSE] status=200 render=index.html")
    return render_template("index.html")


@app.route("/health", methods=["GET"])
def health():
    """서버 상태와 API Key 설정 여부만 반환 (보안을 위해 키 값 자체는 절대 노출 금지)"""
    logger.info("[REQUEST] GET /health")
    
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    has_api_key = bool(api_key and api_key != "your_gemini_api_key_here")
    model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    
    response_data = {
        "status": "ok",
        "has_api_key": has_api_key,
        "model": model_name
    }
    
    logger.info(f"[RESPONSE] status=200 has_api_key={has_api_key} model={model_name}")
    return jsonify(response_data)


@app.route("/inspect", methods=["POST"])
def inspect_file():
    """
    POST /inspect (1차 + 2차 완성):
    1. 파일명, 파일 크기 검증
    2. 인코딩 순차 시도 및 헤더 줄 자동 감지
    3. 행 수, 열 수, 열 이름, 열별 타입
    4. 숫자형/문자형/연도 열 분류
    5. 결측치 수 및 비율, 중복 행 수
    6. Unnamed 열 / 완전히 비어 있는 열
    7. 연도별 데이터 충실도 및 권장 기준연도
    8. 집계 행(대륙/소득그룹 등) 후보 탐지
    9. 데이터 미리보기 10행 (안전한 JSON 직렬화)
    10. 경고 목록 정리
    """
    logger.info("[REQUEST] POST /inspect")
    warnings = []

    try:
        # 1. 파일 업로드 유무 검증
        if "file" not in request.files:
            logger.warning("[ERROR] 파일이 전송되지 않음")
            return jsonify({"error": "업로드된 파일이 없습니다. CSV 파일을 선택해 주세요."}), 400

        file = request.files["file"]
        filename = file.filename or ""

        if not filename:
            logger.warning("[ERROR] 파일명이 비어있음")
            return jsonify({"error": "선택된 파일명이 없습니다."}), 400

        if not filename.lower().endswith(".csv"):
            logger.warning(f"[ERROR] 허용되지 않은 확장자: {filename}")
            return jsonify({"error": "지원하지 않는 파일 형식입니다. CSV(.csv) 파일만 업로드할 수 있습니다."}), 400

        file_bytes = file.read()
        file_size = len(file_bytes)
        logger.info(f"[FILE] name={filename} size={file_size} bytes")

        if file_size == 0:
            logger.warning("[ERROR] 빈 파일 업로드")
            return jsonify({"error": "파일 내용이 비어 있습니다(0바이트)."}), 400

        # 2. 인코딩 순차 시도
        text, detected_encoding = try_detect_encoding(file_bytes)
        if text is None:
            logger.error("[ERROR] 인코딩 감지 실패")
            return jsonify({
                "error": "파일 인코딩을 인식할 수 없습니다. utf-8, utf-8-sig, cp949, euc-kr 인코딩을 지원합니다."
            }), 400

        # 3. 헤더 행 자동 감지
        header_idx, raw_header_cols = detect_header_row(text)
        if header_idx > 0:
            warnings.append(
                f"표 상단에 안내 문구 또는 빈 줄이 있어 상위 {header_idx}개 줄을 건너뛰고 "
                f"{header_idx + 1}번째 줄을 열 이름으로 인식했습니다."
            )

        # 4. Pandas 파싱
        try:
            df = pd.read_csv(io.StringIO(text), skiprows=header_idx)
        except Exception as parse_err:
            logger.error(f"[ERROR] CSV 파싱 실패: {parse_err}\n{traceback.format_exc()}")
            return jsonify({"error": f"CSV 파일 파싱 중 오류가 발생했습니다: {str(parse_err)}"}), 400

        rows_count = len(df)
        cols_count = len(df.columns)
        logger.info(f"[CSV] parsing 완료 rows={rows_count} columns={cols_count}")

        # 5. 열 이름 및 타입 분류
        col_names = [str(c) for c in df.columns]
        col_types = {str(c): str(df[c].dtype) for c in df.columns}

        # 4자리 숫자(1900~2099)로 된 연도 열 감지
        year_cols = [c for c in df.columns if re.match(r"^(19|20)\d{2}$", str(c).strip())]
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c not in year_cols]
        text_cols = [c for c in df.columns if c not in numeric_cols and c not in year_cols]

        # 6. Unnamed 열 및 완전히 비어 있는 열 탐지
        unnamed_cols = [str(c) for c in df.columns if "Unnamed:" in str(c) or str(c).strip() == ""]
        empty_cols = [str(c) for c in df.columns if df[c].isna().all()]

        if unnamed_cols:
            warnings.append(
                f"끝에 쉼표(,) 등으로 인해 이름 없는 빈 열이 감지되었습니다 ({len(unnamed_cols)}개: {', '.join(unnamed_cols[:3])})."
            )
        if empty_cols:
            warnings.append(
                f"데이터가 단 하나도 없는 완전히 비어있는 열이 {len(empty_cols)}개 있습니다."
            )

        # 중복 열 이름 검사
        dup_cols = df.columns[df.columns.duplicated()].tolist()
        if dup_cols:
            warnings.append(f"중복된 열 이름이 감지되었습니다: {', '.join(dup_cols)}")

        # 7. 결측치 및 중복 행 계산
        missing_by_col = {}
        for c in df.columns:
            cnt = int(df[c].isna().sum())
            missing_by_col[str(c)] = {
                "count": cnt,
                "rate": round(cnt / rows_count * 100, 2) if rows_count > 0 else 0
            }

        duplicate_rows_cnt = int(df.duplicated().sum())
        if duplicate_rows_cnt > 0:
            warnings.append(f"완전히 동일한 중복 행이 {duplicate_rows_cnt}개 발견되었습니다.")

        # 8. 연도별 데이터 충실도 및 권장 기준연도
        year_coverage = []
        recommended_year = None
        if year_cols:
            for y in year_cols:
                filled = int(df[y].notna().sum())
                rate = round(filled / rows_count * 100, 2) if rows_count > 0 else 0
                year_coverage.append({
                    "year": str(y),
                    "filled_count": filled,
                    "total_rows": rows_count,
                    "fill_rate": rate
                })

            # 최신 연도 순 정렬 후 충실도 70% 이상인 가장 최근 연도를 권장연도로 채택
            sorted_by_year_desc = sorted(year_coverage, key=lambda x: int(x["year"]), reverse=True)
            for item in sorted_by_year_desc:
                if item["fill_rate"] >= 70.0:
                    recommended_year = item["year"]
                    break

            # 70% 넘는 연도가 없으면 최근 5개 연도 중 가장 많이 채워진 연도 선택
            if not recommended_year and sorted_by_year_desc:
                recommended_year = max(sorted_by_year_desc[:5], key=lambda x: x["filled_count"])["year"]

            # 마지막 연도에 결측치가 심한 경우 경고
            latest_year_stat = sorted_by_year_desc[0]
            if latest_year_stat["fill_rate"] < 50.0:
                warnings.append(
                    f"가장 최근 열({latest_year_stat['year']}년)은 데이터 충실도가 {latest_year_stat['fill_rate']}%로 매우 낮습니다. "
                    f"분석 시 충실도가 높은 권장 기준연도({recommended_year}년) 사용을 권장합니다."
                )

        # 9. 집계 행 후보 탐지 (World, High income, Euro area 등)
        aggregate_candidates = []
        label_col = text_cols[0] if text_cols else (df.columns[0] if len(df.columns) > 0 else None)
        if label_col:
            for idx, val in df[label_col].items():
                val_lower = str(val).lower()
                for kw in AGG_KEYWORDS:
                    if kw in val_lower:
                        aggregate_candidates.append({
                            "index": int(idx),
                            "name": str(val),
                            "matched_keyword": kw
                        })
                        break

            if aggregate_candidates:
                warnings.append(
                    f"개별 국가가 아닌 대륙·소득군 등 집계용 행 후보가 {len(aggregate_candidates)}개 감지되었습니다. "
                    f"국가 비교 시 이 행들을 제외해야 왜곡을 방지할 수 있습니다."
                )

        # 10. 데이터 미리보기 10행 (NaN, Inf -> None 치환)
        preview_raw = df.head(10).to_dict(orient="records")
        preview = sanitize_for_json(preview_raw)

        logger.info(f"[DIAGNOSE] 진단 2차 완료 warnings={len(warnings)} agg_candidates={len(aggregate_candidates)}")

        # 11. 메모리 데이터셋에 적재
        global CURRENT_DATASET
        diagnosis_result = {
            "filename": filename,
            "file_size": file_size,
            "file_size_human": f"{file_size / 1024:.2f} KB" if file_size >= 1024 else f"{file_size} B",
            "encoding": detected_encoding,
            "skipped_rows": header_idx,
            "rows": rows_count,
            "columns": cols_count,
            "column_names": col_names,
            "column_types": col_types,
            "numeric_columns": [str(c) for c in numeric_cols],
            "text_columns": [str(c) for c in text_cols],
            "year_columns": [str(c) for c in year_cols],
            "missing_values": missing_by_col,
            "duplicate_rows": duplicate_rows_cnt,
            "unnamed_columns": unnamed_cols,
            "empty_columns": empty_cols,
            "year_coverage": year_coverage,
            "recommended_year": recommended_year,
            "aggregate_candidates": aggregate_candidates,
            "aggregate_label_column": str(label_col) if label_col else None,
            "preview": preview,
            "warnings": warnings,
            "memory_notice": "파일은 서버 디스크에 저장되지 않으며, 서버를 재시작하면 메모리에서 자동으로 사라집니다."
        }

        CURRENT_DATASET.clear()
        CURRENT_DATASET.update({
            "df": df,
            "filename": filename,
            "encoding": detected_encoding,
            "skipped_rows": header_idx,
            "diagnosis": diagnosis_result,
            "last_run_result": None,
            "last_explanation": "",
            "last_explain_mode": "B"
        })

        logger.info(f"[RESPONSE] status=200 diagnosis_complete=true warnings={len(warnings)}")
        return jsonify(sanitize_for_json({"status": "success", "diagnosis": diagnosis_result}))

    except Exception as e:
        logger.error(f"[ERROR] inspect 도중 예외 발생: {str(e)}\n{traceback.format_exc()}")
        return jsonify({"error": f"서버 내부 오류가 발생했습니다: {str(e)}"}), 500


@app.route("/suggest", methods=["POST"])
def suggest_analyses():
    """
    POST /suggest:
    진단 결과 요약과 12종 도구 카탈로그를 Gemini API에 전달하여
    실행 가능한 최적의 분석 3~5개를 JSON 형식으로 추천받아 반환.
    """
    logger.info("[REQUEST] POST /suggest")

    try:
        global CURRENT_DATASET
        if CURRENT_DATASET["df"] is None or CURRENT_DATASET["diagnosis"] is None:
            logger.warning("[ERROR] 진단된 데이터셋이 없음")
            return jsonify({
                "error": "먼저 CSV 파일을 업로드하고 진단을 완료해야 분석 제안을 받을 수 있습니다."
            }), 400

        api_key = os.getenv("GEMINI_API_KEY", "").strip()
        if not api_key or api_key == "your_gemini_api_key_here":
            logger.warning("[ERROR] GEMINI_API_KEY 미설정")
            return jsonify({
                "error": "Gemini API Key가 설정되지 않았습니다. .env 파일에 GEMINI_API_KEY를 설정해 주세요."
            }), 400

        model_name = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")
        diag = CURRENT_DATASET["diagnosis"]

        # CSV 원본 전체를 보내지 않고, 진단 요약만 추출하여 전송 (절대 규칙 준수)
        diagnosis_summary = {
            "filename": diag["filename"],
            "rows": diag["rows"],
            "columns": diag["columns"],
            "text_columns": diag["text_columns"],
            "year_columns_sample": diag["year_columns"][:5] + ["..."] + diag["year_columns"][-5:] if len(diag["year_columns"]) > 10 else diag["year_columns"],
            "year_range": f"{diag['year_columns'][0]}~{diag['year_columns'][-1]}" if diag["year_columns"] else "없음",
            "recommended_year": diag["recommended_year"],
            "warnings": diag["warnings"],
            "aggregate_candidates_count": len(diag["aggregate_candidates"]),
            "preview_sample": diag["preview"][:3]
        }

        catalog_info = [
            {
                "id": t["id"],
                "name": t["name"],
                "description": t["description"],
                "requires_reference_period": t["requires_reference_period"],
                "chart_type": t["chart_type"]
            }
            for t in TOOL_CATALOG
        ]

        system_instruction = f"""
당신은 통계 및 데이터 분석 수석 전문가입니다.
사용자가 업로드한 데이터의 진단 요약 정보와 실행 가능한 12종 도구 카탈로그를 검토하고,
이 데이터셋에서 가장 인사이트가 크고 유의미한 분석 3~5개를 골라 제안하십시오.

[절대 규칙]
1. 반드시 아래 도구 카탈로그 목록에 명시된 ID의 도구만 제안하십시오. (카탈로그 외 도구 제안 절대 금지)
2. 데이터셋에 실제로 존재하는 열 이름만 파라미터로 제안하십시오.
3. 기준 시점(reference_period)이 필요한 도구는 데이터의 권장 기준연도('{diag["recommended_year"]}')를 우선 활용하십시오.
4. 직접 계산하거나 임의의 숫자를 지어내지 마십시오. 당신의 역할은 '분석 제안'이며, 실제 계산은 Pandas가 수행합니다.
5. 반드시 아래 JSON 형식만을 출력하십시오.

[도구 카탈로그]
{json.dumps(catalog_info, ensure_ascii=False, indent=2)}

[출력 JSON 형식]
{{
  "suggestions": [
    {{
      "tool": "도구ID (카탈로그 id 중 하나)",
      "name": "도구 한글 명칭",
      "params": {{
        "reference_period": "{diag['recommended_year']}",
        "exclude_aggregates": true
      }},
      "reference_period": "{diag['recommended_year']}",
      "why": "이 분석이 이 데이터에서 의미 있는 구체적인 이유",
      "caution": "해석할 때 사용자가 반드시 주의해야 할 점"
    }}
  ]
}}
"""

        user_prompt = f"""
다음은 업로드된 데이터의 진단 결과 요약입니다:
{json.dumps(diagnosis_summary, ensure_ascii=False, indent=2)}

위 데이터를 바탕으로 도구 카탈로그에서 3~5개의 최적 분석을 골라 제안 JSON을 작성해주세요.
"""

        logger.info(f"[AI] suggest 요청 시작 (model={model_name})")
        client = genai.Client(api_key=api_key)

        try:
            ai_resp = client.models.generate_content(
                model=model_name,
                contents=system_instruction + "\n\n" + user_prompt,
                config=types.GenerateContentConfig(response_mime_type="application/json")
            )
            raw_text = ai_resp.text or ""
            logger.info("[AI] suggest 응답 수신 완료")
            parsed_json = json.loads(raw_text)
            suggestions = parsed_json.get("suggestions", [])
            logger.info(f"[AI] JSON 파싱 완료 (제안 개수: {len(suggestions)})")
        except Exception as ai_err:
            logger.error(f"[ERROR] Gemini 호출 또는 JSON 파싱 실패: {ai_err}\n{traceback.format_exc()}")
            return jsonify({
                "error": f"AI 분석 제안 생성 중 오류가 발생했습니다: {str(ai_err)}"
            }), 502

        valid_suggestions = []
        catalog_ids = {t["id"]: t["name"] for t in TOOL_CATALOG}

        for item in suggestions:
            tool_id = item.get("tool")
            if tool_id in catalog_ids:
                if not item.get("name"):
                    item["name"] = catalog_ids[tool_id]
                if not item.get("params"):
                    item["params"] = {}
                if tool_id in ["top_bottom_n", "group_summary", "change_rate", "distribution", "correlation_scatter", "compare_one_vs_all"]:
                    if "reference_period" not in item["params"]:
                        item["params"]["reference_period"] = item.get("reference_period") or diag["recommended_year"]
                    item["reference_period"] = item["params"]["reference_period"]
                if "exclude_aggregates" not in item["params"]:
                    item["params"]["exclude_aggregates"] = True
                valid_suggestions.append(item)

        if len(valid_suggestions) < 3:
            rec_y = diag["recommended_year"] or "2024"
            fallback_tools = [
                {
                    "tool": "top_bottom_n",
                    "name": "기준연도 상위·하위 N 순위",
                    "params": {"reference_period": rec_y, "n": 10, "exclude_aggregates": True},
                    "reference_period": rec_y,
                    "why": f"{rec_y}년 시점에 데이터가 완전한 국가들을 대상으로 상위 및 하위 10개국 순위를 비교합니다.",
                    "caution": "동일 시점 기준 비교이므로 결측 국가가 일부 제외될 수 있습니다."
                },
                {
                    "tool": "trend_line",
                    "name": "시계열 추이 비교 (선 차트)",
                    "params": {"items": ["Korea, Rep.", "United States", "Japan", "World"], "exclude_aggregates": False},
                    "reference_period": "1990~2024",
                    "why": "주요 국가와 글로벌 평균의 연도별 인터넷 보급률 변화 추세를 비교합니다.",
                    "caution": "국가별 결측 연도가 있는 구간은 선이 이어지지 않을 수 있습니다."
                },
                {
                    "tool": "distribution",
                    "name": "구간별 빈도 분포 (히스토그램)",
                    "params": {"reference_period": rec_y, "bins": 10, "exclude_aggregates": True},
                    "reference_period": rec_y,
                    "why": f"{rec_y}년 전 세계 국가들의 인터넷 사용률이 어느 구간에 집중되어 있는지 분포를 분석합니다.",
                    "caution": "상위 구간에 국가가 집중되어 있어 정규분포가 아닐 수 있습니다."
                }
            ]
            existing_tools = {s["tool"] for s in valid_suggestions}
            for fb in fallback_tools:
                if fb["tool"] not in existing_tools and len(valid_suggestions) < 4:
                    valid_suggestions.append(fb)

        logger.info(f"[RESPONSE] status=200 suggestions_count={len(valid_suggestions)}")
        return jsonify({
            "status": "success",
            "suggestions": valid_suggestions
        })

    except Exception as e:
        logger.error(f"[ERROR] suggest 도중 예외 발생: {str(e)}\n{traceback.format_exc()}")
        return jsonify({"error": f"서버 내부 오류가 발생했습니다: {str(e)}"}), 500


@app.route("/run", methods=["POST"])
def run_tool():
    """
    POST /run:
    선택된 분석 도구를 Pandas로 직접 실행하여 수치 계산 결과 및 Chart.js 차트 데이터를 반환.
    - 7~12번 도구는 기준 시점(reference_period) 필수 검증
    - 계산 결과, 기준시점, 사용 행 수, 제외 행 수와 이유, 주의사항 반환
    - 직전 실행 결과를 CURRENT_DATASET["last_run_result"]에 보관 (이후 /explain 모드 B에서 사용)
    """
    logger.info("[REQUEST] POST /run")

    try:
        global CURRENT_DATASET
        if CURRENT_DATASET["df"] is None:
            logger.warning("[ERROR] 실행할 데이터셋이 메모리에 없음")
            return jsonify({
                "error": "업로드된 데이터셋이 없습니다. 먼저 CSV 파일을 업로드하고 진단을 완료해주세요."
            }), 400

        req_data = request.get_json(silent=True) or {}
        tool_name = req_data.get("tool")
        params = req_data.get("params", {})

        if "reference_period" in req_data and "reference_period" not in params:
            params["reference_period"] = req_data["reference_period"]

        if not tool_name:
            logger.warning("[ERROR] tool_name 누락")
            return jsonify({"error": "실행할 분석 도구(tool)를 지정해야 합니다."}), 400

        if tool_name not in TOOL_REGISTRY:
            logger.warning(f"[ERROR] 등록되지 않은 도구: {tool_name}")
            return jsonify({
                "error": f"도구 목록에 존재하지 않는 도구입니다: '{tool_name}'. 지원 도구: {', '.join(TOOL_REGISTRY.keys())}"
            }), 400

        tool_fn = TOOL_REGISTRY[tool_name]

        tools_requiring_ref_period = [
            "top_bottom_n", "group_summary", "change_rate",
            "distribution", "correlation_scatter", "compare_one_vs_all"
        ]
        if tool_name in tools_requiring_ref_period:
            ref_val = params.get("reference_period")
            if not ref_val or str(ref_val).strip() == "":
                rec_year = (CURRENT_DATASET.get("diagnosis") or {}).get("recommended_year")
                if rec_year:
                    params["reference_period"] = rec_year
                else:
                    return jsonify({
                        "error": f"'{tool_name}' 도구는 기준 시점(reference_period) 파라미터가 필수입니다. (예: '2024')"
                    }), 400

        sig = inspect.signature(tool_fn)
        valid_keys = set(sig.parameters.keys()) - {"df"}
        filtered_params = {k: v for k, v in params.items() if k in valid_keys}

        df = CURRENT_DATASET["df"]

        result_obj = tool_fn(df, **filtered_params)

        logger.info(
            f"[TOOL] tool={tool_name} params={filtered_params} "
            f"used_rows={result_obj.get('used_rows')} excluded_rows={result_obj.get('excluded_rows')}"
        )

        CURRENT_DATASET["last_run_result"] = result_obj

        logger.info("[RESPONSE] status=200 tool_run_success=true")
        return jsonify({
            "status": "success",
            "data": result_obj
        })

    except ValueError as ve:
        logger.warning(f"[ERROR] 파라미터 검증 실패: {ve}")
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        logger.error(f"[ERROR] 도구 실행 도중 예외 발생: {str(e)}\n{traceback.format_exc()}")
        return jsonify({"error": f"도구 실행 중 오류가 발생했습니다: {str(e)}"}), 500


def get_indicator_metadata():
    """data 폴더 내의 Metadata_Indicator CSV 파일이 있으면 지표명, 정의, 출처를 추출"""
    ind_files = glob.glob("data/**/Metadata_Indicator*.csv", recursive=True)
    if ind_files:
        try:
            m_df = pd.read_csv(ind_files[0], encoding="utf-8-sig")
            if not m_df.empty:
                return {
                    "name": str(m_df.iloc[0].get("INDICATOR_NAME", "")),
                    "note": str(m_df.iloc[0].get("SOURCE_NOTE", "")),
                    "source": str(m_df.iloc[0].get("SOURCE_ORGANIZATION", ""))
                }
        except Exception:
            pass
    return {
        "name": "인터넷 이용자 비율 (% of population)",
        "note": "지난 3개월 동안 모든 장소에서 컴퓨터, 스마트폰 등을 이용해 인터넷을 1회 이상 사용한 인구 비율",
        "source": "국제전기통신연합(ITU) 및 세계은행(World Bank)"
    }


@app.route("/explain", methods=["POST"])
def explain_result():
    """
    POST /explain:
    Gemini를 통한 분석 결과 해석 라우트 (두 가지 모드 지원).
    - 모드 A (비교용, 의도적으로 허술하게):
      진단 요약만 전달하고 "이 데이터를 분석해서 보고서를 써줘"라고 요청. Pandas 계산 결과 미제공.
    - 모드 B (정식):
      Pandas가 계산한 결과(표, 기준시점, 제외 행 수)만 전달하고 8대 엄격 규칙 및 7개 헤더 구조로 해석.
    """
    req_data = request.get_json(silent=True) or {}
    mode = req_data.get("mode", "B").upper()
    if mode not in ["A", "B"]:
        mode = "B"

    logger.info(f"[REQUEST] POST /explain (mode={mode})")

    try:
        global CURRENT_DATASET
        if CURRENT_DATASET["df"] is None or CURRENT_DATASET["diagnosis"] is None:
            logger.warning("[ERROR] 진단된 데이터셋이 없음")
            return jsonify({
                "error": "업로드된 데이터셋이 없습니다. 먼저 CSV 파일을 업로드하고 진단을 완료해주세요."
            }), 400

        api_key = os.getenv("GEMINI_API_KEY", "").strip()
        if not api_key or api_key == "your_gemini_api_key_here":
            logger.warning("[ERROR] GEMINI_API_KEY 미설정")
            return jsonify({
                "error": "Gemini API Key가 설정되지 않았습니다. .env 파일에 GEMINI_API_KEY를 설정해 주세요."
            }), 400

        model_name = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")
        client = genai.Client(api_key=api_key)
        diag = CURRENT_DATASET["diagnosis"]
        ind_meta = get_indicator_metadata()

        if mode == "A":
            diag_summary = {
                "filename": diag["filename"],
                "rows": diag["rows"],
                "columns": diag["columns"],
                "text_columns": diag["text_columns"],
                "year_range": f"{diag['year_columns'][0]}~{diag['year_columns'][-1]}" if diag["year_columns"] else "없음",
                "recommended_year": diag["recommended_year"]
            }

            prompt_a = f"""
다음은 업로드된 데이터 파일의 기본 정보입니다:
{json.dumps(diag_summary, ensure_ascii=False, indent=2)}

이 데이터를 바탕으로 국가별 인터넷 사용 현황에 대해 자유롭게 분석 보고서를 써주세요.
"""
            logger.info("[AI] explain 요청 시작 (모드 A)")
            ai_resp = client.models.generate_content(
                model=model_name,
                contents=prompt_a
            )
            explanation_text = ai_resp.text or ""
            logger.info("[AI] explain 응답 수신 (모드 A)")

        else:
            calc_result = req_data.get("calculation_result") or CURRENT_DATASET.get("last_run_result")
            if not calc_result:
                logger.warning("[ERROR] 계산 결과 없음 (모드 B)")
                return jsonify({
                    "error": "정식 모드(모드 B) 해석을 위해서는 먼저 분석 도구를 실행(/run)하여 계산 결과를 생성해야 합니다."
                }), 400

            system_instruction_b = f"""
당신은 엄격하고 정직한 시니어 데이터 분석 보고관입니다.
반드시 아래 제공된 [Pandas 실제 계산 결과]와 [지표 메타데이터]에 기반해서만 분석 보고서를 작성하십시오.

[엄격한 8대 규칙 - 절대 위반 금지]
1. 제공된 계산 결과에 있는 숫자만 사용하십시오. 직접 계산하거나 추정하지 마십시오.
2. 데이터에 없는 국가, 연도, 열을 임의로 지어내거나 언급하지 마십시오.
3. 기준 시점과 제외된 행 수, 그 이유를 반드시 본문에 명시하십시오.
4. 상관관계를 인과관계로 표현하지 마십시오.
5. 지표의 정의와 출처를 보고서 앞부분에 명확히 표기하십시오.
6. 확인할 수 없는 것은 "이 데이터로는 확인할 수 없다"라고 정직하게 쓰십시오.
7. 중학생도 이해할 수 있게 쉬운 문장으로 쓰되, 정확성을 절대 희생하지 마십시오.
8. 반드시 아래 명시된 7개의 마크다운 헤더(##)를 빠짐없이 순서대로 사용하십시오.

[지표 메타데이터]
- 지표명: {ind_meta.get('name')}
- 지표 정의: {ind_meta.get('note')}
- 데이터 출처: {ind_meta.get('source')}

[Pandas 실제 계산 결과]
{json.dumps(calc_result, ensure_ascii=False, indent=2)}

[출력 형식 - 반드시 아래 7개 제목을 그대로 사용하십시오]
## 무엇을 봤는가
## 데이터 출처와 정의
## 기준 시점과 제외된 항목
## 계산 결과
## 읽어낼 수 있는 것
## 이 데이터로는 알 수 없는 것
## 다음에 확인해볼 것
"""

            user_prompt_b = "위의 계산 결과와 지표 메타데이터를 정밀하게 분석하여 7개 항목의 정식 보고서를 작성해주세요."

            logger.info("[AI] explain 요청 시작 (모드 B)")
            ai_resp = client.models.generate_content(
                model=model_name,
                contents=system_instruction_b + "\n\n" + user_prompt_b
            )
            explanation_text = ai_resp.text or ""
            logger.info("[AI] explain 응답 수신 (모드 B)")

        # 생성된 보고서 원문과 모드를 메모리에 보관하여 후속 질문(/chat)에서 참조할 수 있도록 함
        CURRENT_DATASET["last_explanation"] = explanation_text
        CURRENT_DATASET["last_explain_mode"] = mode

        logger.info(f"[RESPONSE] status=200 mode={mode}")
        return jsonify({
            "status": "success",
            "mode": mode,
            "explanation": explanation_text
        })

    except Exception as e:
        logger.error(f"[ERROR] explain 도중 예외 발생: {str(e)}\n{traceback.format_exc()}")
        return jsonify({"error": f"분석 해석 생성 중 오류가 발생했습니다: {str(e)}"}), 500


@app.route("/chat", methods=["POST"])
def chat_followup():
    """
    POST /chat:
    최종 보고서 및 Pandas 계산 결과를 바탕으로 후속 질문에 대해
    중학생도 단번에 이해할 수 있는 친절하고 쉬운 비유로 답변을 제공하는 대화 엔드포인트.
    """
    logger.info("[REQUEST] POST /chat")
    try:
        global CURRENT_DATASET
        if CURRENT_DATASET["df"] is None or CURRENT_DATASET["diagnosis"] is None:
            logger.warning("[ERROR] 진단된 데이터셋이 없음")
            return jsonify({
                "error": "업로드된 데이터셋이 없습니다. 먼저 CSV 파일을 업로드하고 진단을 완료해주세요."
            }), 400

        req_data = request.get_json(silent=True) or {}
        user_message = (req_data.get("message") or "").strip()
        chat_history = req_data.get("chat_history") or []
        mode = req_data.get("mode", CURRENT_DATASET.get("last_explain_mode", "B")).upper()

        if not user_message:
            logger.warning("[ERROR] 질문 내용 누락")
            return jsonify({"error": "질문 내용을 입력해주세요."}), 400

        api_key = os.getenv("GEMINI_API_KEY", "").strip()
        if not api_key or api_key == "your_gemini_api_key_here":
            logger.warning("[ERROR] GEMINI_API_KEY 미설정")
            return jsonify({"error": "Gemini API 키가 설정되지 않았습니다."}), 500

        model_name = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")
        diag = CURRENT_DATASET.get("diagnosis") or {}
        last_result = CURRENT_DATASET.get("last_run_result") or {}
        last_exp = CURRENT_DATASET.get("last_explanation", "")
        ind_meta = get_indicator_metadata()

        system_instruction = f"""
당신은 복잡하고 어려운 데이터 통계 보고서를 누구나, 특히 **중학생(만 13~15세 청소년)도 단번에 이해할 수 있도록 쉽고 친절하게 설명해주는 최고의 AI 데이터 선생님**입니다.

[상황 및 분석 컨텍스트]
- 현재 분석 모드: 모드 {mode} ({'엄격한 팩트 기반' if mode == 'B' else '자유 추정 모드'})
- 데이터 파일명: {diag.get('filename')}
- 지표 정보: {ind_meta.get('name')} (출처: {ind_meta.get('source')})
- 지표 정의: {ind_meta.get('note')}
- Pandas 계산 결과 요약:
{json.dumps(last_result, ensure_ascii=False, indent=2) if last_result else "계산 결과 없음"}
- 최근 생성된 분석 보고서 원문:
{last_exp if last_exp else "보고서 없음"}

[선생님으로서 지켜야 할 5대 원칙]
1. **중학생 눈높이 맞춤 (친숙한 일상 비유 필수 활용)**:
   - "인터넷 보급률 100%"는 "한 반에 30명 학생이 있다면 30명 모두가 스마트폰이나 컴퓨터로 인터넷을 쓰고 있는 것과 같아요"처럼 친숙한 학교/일상 비유로 설명하세요.
   - 어려운 통계 용어(결측치, 중앙값, 사분위수, 집계 행 등)가 나오면 반드시 괄호나 쉬운 일상어로 풀어주세요.
     * 결측치: 조사가 안 되어 장부에 빈칸으로 남은 나라
     * 집계 행: 개별 국가가 아니라 '전 세계 평균'이나 '고소득 국가 모임'처럼 여러 나라를 묶어놓은 그룹
     * 중앙값: 모든 나라를 1등부터 줄 세웠을 때 딱 한가운데 서 있는 나라의 점수
2. **숫자 날조(환각) 절대 금지**:
   - 절대로 가짜 통계나 숫자를 지어내지 마세요. 주어진 Pandas 계산 결과와 보고서에 있는 수치(예: 193개국, 제외된 72개 행, 상위 국가 수치 등)만을 팩트로 인용하세요.
3. **친절하고 격려하는 말투**:
   - 친절하고 다정한 '해요체'(~해요, ~랍니다, ~맞아요!)를 사용하세요.
4. **시각적으로 편안한 구성**:
   - 긴 줄글보다는 핵심 포인트를 불릿 기호(`-`), 이모지(🎒, 💡, 📊, 🌍 등)로 나누어 읽기 쉽게 정리해 주세요.
5. **모드 {mode} 존중**:
   - 모드 B인 경우: 팩트와 수치적 정확성을 철저히 지키며 쉬운 언어로 풀어주세요.
   - 모드 A인 경우: "이 보고서는 AI가 추정한 내용이 일부 섞여 있으니 감안해서 들어주세요"라고 부드럽게 짚어주세요.
"""

        history_prompts = []
        for h in chat_history[-6:]:
            role_label = "학생(사용자)" if h.get("role") == "user" else "AI 선생님"
            history_prompts.append(f"{role_label}: {h.get('content')}")

        conversation_context = "\n".join(history_prompts)
        user_prompt = f"""
[이전 대화 내역]
{conversation_context if conversation_context else "이전 대화 없음"}

[학생의 새로운 질문]
{user_message}

위 질문에 대해 중학생도 무릎을 탁 치며 이해할 수 있도록 쉽고 명쾌하며 친절하게 답변해 주세요!
"""

        logger.info(f"[AI] chat 요청 시작 (model={model_name})")
        client = genai.Client(api_key=api_key)
        ai_resp = client.models.generate_content(
            model=model_name,
            contents=system_instruction + "\n\n" + user_prompt
        )
        reply_text = ai_resp.text or "답변을 생성하지 못했습니다."
        logger.info("[AI] chat 응답 수신 완료")

        return jsonify({
            "status": "success",
            "reply": reply_text
        })

    except Exception as e:
        logger.error(f"[ERROR] chat 도중 예외 발생: {str(e)}\n{traceback.format_exc()}")
        return jsonify({"error": f"답변 생성 중 오류가 발생했습니다: {str(e)}"}), 500


@app.route("/ask", methods=["POST"])
def ask_custom_query():
    """
    POST /ask:
    사용자의 자연어 질문(예: '2024년 기준 인터넷 이용률 상위 10개 국가와 한국의 위치를 알려줘')을 받아
    1. Gemini가 질문에서 분석 대상 연도, 순위 N, 대상 국가(한국 등)를 구조화된 의도로 추출
    2. Pandas가 100% 직접 순위, 수치, 격차, 표 데이터를 정확히 계산 (환각 방지)
    3. Gemini가 계산된 팩트 수치를 바탕으로 중학생 눈높이의 친절하고 쉬운 비유 해설을 작성
    """
    logger.info("[REQUEST] POST /ask")
    try:
        global CURRENT_DATASET
        if CURRENT_DATASET["df"] is None or CURRENT_DATASET["diagnosis"] is None:
            logger.warning("[ERROR] 업로드된 데이터셋이 없음")
            return jsonify({
                "error": "업로드된 데이터셋이 없습니다. 먼저 CSV 파일을 업로드하고 진단을 완료해주세요."
            }), 400

        req_data = request.get_json(silent=True) or {}
        question = (req_data.get("question") or "").strip()
        if not question:
            return jsonify({"error": "질문 내용을 입력해주세요."}), 400

        api_key = os.getenv("GEMINI_API_KEY", "").strip()
        if not api_key or api_key == "your_gemini_api_key_here":
            logger.warning("[ERROR] GEMINI_API_KEY 미설정")
            return jsonify({"error": "Gemini API 키가 설정되지 않았습니다."}), 500

        model_name = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")
        client = genai.Client(api_key=api_key)
        df = CURRENT_DATASET["df"]
        diag = CURRENT_DATASET["diagnosis"]
        year_cols = diag.get("year_columns", [])
        rec_year = diag.get("recommended_year", "2024")
        label_col = diag.get("aggregate_label_column") or (diag.get("text_columns", ["Country Name"])[0])

        # 1. Gemini를 통해 질문의 의도 및 파라미터 파싱
        parse_instruction = f"""
당신은 사용자의 질문에서 데이터 분석에 필요한 정확한 파라미터를 추출하는 파서(Parser)입니다.
현재 데이터셋 정보:
- 사용 가능한 연도 목록: {year_cols[-10:] if len(year_cols) > 10 else year_cols} (권장: '{rec_year}')
- 국가/항목 열 이름: '{label_col}'

사용자의 질문을 분석하여 아래 JSON 규격으로만 응답하세요:
{{
  "year": "질문에서 언급된 연도(4자리 숫자, 없으면 '{rec_year}')",
  "top_n": 10,
  "target_country_query": "비교하고 싶은 특정 국가명 (예: 한국, 대한민국 -> 'Korea, Rep.' 또는 'Korea', 없으면 null)",
  "sort_order": "desc"
}}
"""
        logger.info(f"[AI] ask 쿼리 의도 파싱 시작: '{question}'")
        parse_resp = client.models.generate_content(
            model=model_name,
            contents=parse_instruction + f"\n\n[사용자 질문]\n{question}",
            config=types.GenerateContentConfig(response_mime_type="application/json")
        )
        parsed_intent = json.loads(parse_resp.text or "{}")
        logger.info(f"[AI] ask 의도 파싱 완료: {parsed_intent}")

        target_year = str(parsed_intent.get("year") or rec_year)
        if target_year not in df.columns:
            target_year = rec_year

        try:
            top_n = int(parsed_intent.get("top_n") or 10)
        except Exception:
            top_n = 10
        target_country_q = parsed_intent.get("target_country_query")
        sort_order = parsed_intent.get("sort_order", "desc")

        # 2. Pandas를 통한 100% 정확한 수치 계산
        filtered_df, excl_cnt, excl_reason = filter_aggregates_if_needed(df, exclude_aggregates=True, label_col=label_col)
        
        num_series = pd.to_numeric(filtered_df[target_year], errors="coerce")
        valid_mask = num_series.notna()
        valid_df = filtered_df[valid_mask].copy()
        valid_df["_val"] = num_series[valid_mask]
        
        ascending = (sort_order == "asc")
        valid_df.sort_values(by="_val", ascending=ascending, inplace=True)
        valid_df.reset_index(drop=True, inplace=True)
        valid_df["_rank"] = valid_df.index + 1

        total_valid_countries = len(valid_df)
        mean_val = round(float(valid_df["_val"].mean()), 2)
        median_val = round(float(valid_df["_val"].median()), 2)

        # 상위 top_n 국가 추출
        top_slice = valid_df.head(top_n)
        top_list = []
        for _, r in top_slice.iterrows():
            top_list.append({
                "rank": int(r["_rank"]),
                "country": str(r[label_col]),
                "value": round(float(r["_val"]), 2),
                "is_target": False
            })

        # 특정 타깃 국가(예: 한국) 검색 및 위치 파악
        target_info = None
        if target_country_q or any(kw in question for kw in ["한국", "대한민국", "korea"]):
            t_query = str(target_country_q or "korea").lower()
            if any(k in t_query or k in question for k in ["korea", "한국", "대한민국", "south korea"]):
                match_mask = valid_df[label_col].astype(str).str.contains("Korea, Rep.|Korea", case=False, regex=True)
            else:
                match_mask = valid_df[label_col].astype(str).str.contains(t_query, case=False)

            matched_rows = valid_df[match_mask]
            if not matched_rows.empty:
                t_row = matched_rows.iloc[0]
                t_rank = int(t_row["_rank"])
                t_val = round(float(t_row["_val"]), 2)
                t_pct = round((t_rank / total_valid_countries) * 100, 1)
                
                # 상위 1위와의 격차, 10위와의 격차 계산
                rank1_val = top_list[0]["value"] if top_list else t_val
                rank10_val = top_list[-1]["value"] if len(top_list) >= top_n else t_val
                gap_with_1st = round(rank1_val - t_val, 2)
                gap_with_10th = round(rank10_val - t_val, 2)

                target_info = {
                    "country": str(t_row[label_col]),
                    "rank": t_rank,
                    "value": t_val,
                    "total_countries": total_valid_countries,
                    "top_percentage": t_pct,
                    "gap_with_1st": gap_with_1st,
                    "gap_with_10th": gap_with_10th,
                    "gap_with_mean": round(t_val - mean_val, 2)
                }

                # 만약 top_list 안에 target 국가가 이미 있으면 플래그 설정
                in_top = False
                for item in top_list:
                    if item["country"] == target_info["country"]:
                        item["is_target"] = True
                        in_top = True
                        break
                if not in_top:
                    target_info["in_top_list"] = False

        # 표(Table) 데이터 생성
        table_rows = []
        for item in top_list:
            table_rows.append({
                "rank": item["rank"],
                "country": item["country"],
                "value": f"{item['value']}%",
                "is_target": item["is_target"],
                "note": "[비교 대상: 한국]" if item["is_target"] else ""
            })

        if target_info and not any(r["is_target"] for r in table_rows):
            # 10위 밖인 경우 구분선과 함께 한국 행 추가
            table_rows.append({
                "rank": "...",
                "country": "...",
                "value": "...",
                "is_target": False,
                "note": ""
            })
            table_rows.append({
                "rank": target_info["rank"],
                "country": target_info["country"],
                "value": f"{target_info['value']}%",
                "is_target": True,
                "note": f"[한국] 전체 {total_valid_countries}개국 중 {target_info['rank']}위 (상위 {target_info['top_percentage']}%)"
            })

        # 3. Chart.js 시각화용 차트 데이터 생성 (가로 막대 그래프)
        chart_labels = []
        chart_data = []
        chart_bg = []
        for item in top_list:
            chart_labels.append(item["country"])
            chart_data.append(item["value"])
            chart_bg.append("rgba(234, 179, 8, 0.9)" if item["is_target"] else "rgba(54, 162, 235, 0.7)")

        if target_info and not any(item["is_target"] for item in top_list):
            chart_labels.extend(["---", f"{target_info['country']} (#{target_info['rank']})"])
            chart_data.extend([None, target_info["value"]])
            chart_bg.extend(["rgba(0,0,0,0)", "rgba(239, 68, 68, 0.9)"])

        chart_config = {
            "type": "bar",
            "indexAxis": "y",
            "labels": chart_labels,
            "datasets": [{
                "label": f"{target_year}년 인터넷 이용률 (%)",
                "data": chart_data,
                "backgroundColor": chart_bg
            }]
        }

        # 계산 팩트 요약
        calculated_facts = {
            "year": target_year,
            "total_valid_countries": total_valid_countries,
            "excluded_rows": excl_cnt,
            "top_n": top_n,
            "top_list": top_list,
            "target_country_info": target_info,
            "global_mean": mean_val,
            "global_median": median_val
        }

        # 3단계(Section 3)와 완벽히 호환되는 run_result 객체 생성
        total_rows_all = len(df)
        total_excluded_cnt = total_rows_all - total_valid_countries
        run_result_data = {
            "tool": "custom_query",
            "reference_period": f"{target_year}년",
            "used_rows": total_valid_countries,
            "excluded_rows": total_excluded_cnt,
            "exclusion_reason": f"대륙·소득군 집계 행 {excl_cnt}개 및 {target_year}년 결측치 {total_excluded_cnt - excl_cnt}개 행 제외",
            "caution": f"오직 {target_year}년 시점에 수치가 존재하는 193개국만을 기준으로 순위를 매겼습니다.",
            "result": {
                "title": f"맞춤 질의: {question}",
                "chart": chart_config,
                "table_rows": table_rows,
                "summary": calculated_facts
            }
        }

        # 3단계 및 4단계에서 이어서 사용할 수 있도록 메모리에 캐싱
        CURRENT_DATASET["last_run_result"] = run_result_data

        # 4. Gemini가 계산된 팩트만을 인용하여 중학생 눈높이 해설 생성
        explain_instruction = f"""
당신은 데이터 통계 결과를 **중학생(만 13~15세)도 단번에 이해할 수 있게 친절하고 재미있게 설명해주는 AI 데이터 선생님**입니다.

[사용자 질문]
"{question}"

[Pandas가 100% 직접 계산한 정확한 수치 팩트]
{json.dumps(calculated_facts, ensure_ascii=False, indent=2)}

[답변 작성 4대 규칙]
1. **수치 절대 날조 금지**:
   - 위 JSON에 명시된 숫자(연도 {target_year}, 전체 {total_valid_countries}개국, 순위, 보급률 수치, 격차 등)만을 정확하게 인용하세요.
2. **중학생 맞춤 친숙한 비유 필수 사용**:
   - 예: "193명이 있는 학교에서 한국은 13등을 한 셈이에요!", "100점 만점에 97.9점을 받았고, 10위 나라와는 시험 점수 0.31점 차이밖에 나지 않는 아주 높은 수준이에요!"
3. **핵심 3단 구조로 답변**:
   - **📌 한 줄 요약**: 질문에 대한 직관적인 핵심 결론
   - **🏆 1위~10위 국가들의 특징**: 최상위권(UAE, 바레인, 사우디 등 100% 그룹과 북유럽 국가들)의 모습
   - **🇰🇷 한국의 위치와 의미**: 한국의 순위({target_info['rank'] if target_info else 'N/A'}위, {target_info['value'] if target_info else 'N/A'}%), 전 세계 평균({mean_val}%)과의 압도적인 차이, 10위권과의 격차를 쉬운 일상어로 해설
4. 친절하고 다정한 '해요체'와 이모지를 활용해 가독성 높게 작성하세요.
"""

        logger.info("[AI] ask 맞춤 해설 생성 시작")
        ai_resp = client.models.generate_content(
            model=model_name,
            contents=explain_instruction
        )
        explanation_text = ai_resp.text or ""
        logger.info("[AI] ask 맞춤 해설 생성 완료")

        return jsonify(sanitize_for_json({
            "status": "success",
            "year": target_year,
            "facts": calculated_facts,
            "table_rows": table_rows,
            "explanation": explanation_text,
            "run_result": run_result_data
        }))

    except Exception as e:
        logger.error(f"[ERROR] ask 도중 예외 발생: {str(e)}\n{traceback.format_exc()}")
        return jsonify({"error": f"맞춤 질의 처리 중 오류가 발생했습니다: {str(e)}"}), 500


if __name__ == "__main__":
    debug_mode = os.getenv("FLASK_DEBUG", "true").lower() == "true"
    logger.info(f"AI Data Insight Builder 서버를 시작합니다. (debug={debug_mode})")
    app.run(host="127.0.0.1", port=5000, debug=debug_mode)
