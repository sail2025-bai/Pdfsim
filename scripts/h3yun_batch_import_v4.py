"""
氚云图纸批量入库脚本 v4

核心逻辑:
  1. 调用氚云自定义action getpdffilelist 获取待处理列表
     - SQL已过滤: F0000223为空 + ContentType=application/pdf + 最新附件(max SortKey)
     - 直接返回 fileid + objectid，无需解析附件数组
  2. DownloadBizObjectFile 下载附件(multipart/form-data)
  3. 入库 Pdfsim (BGE 768维向量)
  4. 处理完一律回写 F0000223，确保不再被扫描到:
     - 成功 → doc_id
     - 失败3次(dead) → "入库失败"

使用方式：
  python scripts/h3yun_batch_import.py scan [--limit N]      # 扫描待处理
  python scripts/h3yun_batch_import.py import [--limit N]    # 入库一批
  python scripts/h3yun_batch_import.py daemon [--limit N]    # 持续轮询入库(服务模式)
  python scripts/h3yun_batch_import.py report                # 状态报告
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import List, Optional

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("pdfsim-import")

# ────────────────────── 状态常量 ──────────────────────
STATUS_PENDING = "pending"
STATUS_NO_ATTACHMENT = "no_attachment"
STATUS_NON_PDF = "non_pdf"
STATUS_DOWNLOAD_FAIL = "download_fail"
STATUS_PARSE_FAIL = "parse_fail"
STATUS_INGESTING = "ingesting"
STATUS_DONE = "done"
STATUS_FAILED = "failed"
STATUS_DEAD = "dead"

TERMINAL_STATUSES = {STATUS_NO_ATTACHMENT, STATUS_NON_PDF, STATUS_PARSE_FAIL, STATUS_DEAD}
RETRYABLE_STATUSES = {STATUS_PENDING, STATUS_DOWNLOAD_FAIL, STATUS_FAILED}
STATUS_LABELS = {
    STATUS_PENDING: "待入库", STATUS_NO_ATTACHMENT: "无附件",
    STATUS_NON_PDF: "非PDF附件", STATUS_DOWNLOAD_FAIL: "下载失败",
    STATUS_PARSE_FAIL: "PDF损坏", STATUS_INGESTING: "入库中",
    STATUS_DONE: "已入库", STATUS_FAILED: "入库失败",
    STATUS_DEAD: "重试耗尽(≤3次)",
}


# ────────────────────── 氚云客户端 ──────────────────────
class H3YunClient:
    def __init__(self, engine_code: str, engine_secret: str,
                 base_url: str = "https://www.h3yun.com"):
        self.engine_code = engine_code
        self.engine_secret = engine_secret
        self.base_url = base_url
        self.api_url = f"{base_url}/OpenApi/Invoke"

    def _invoke(self, action: str, **params) -> dict:
        headers = {
            "EngineCode": self.engine_code,
            "EngineSecret": self.engine_secret,
            "Content-Type": "application/json",
        }
        body = {"ActionName": action, **params}
        resp = requests.post(self.api_url, json=body, headers=headers, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        if not data.get("Successful", False):
            raise RuntimeError(f"氚云API错误: {data.get('ErrorMessage', str(data))}")
        return data.get("ReturnData", {})

    def get_pdf_file_list(self, limit: int = 10, controller: str = "",
                          app_code: str = "") -> List[dict]:
        """
        调用自定义action getpdffilelist
        返回: [{"fileid": "xxx", "objectid": "yyy"}, ...]
        SQL逻辑: F0000223为空 + PDF + 最新附件(max SortKey)
        
        自定义action需要额外传 Controller(类名) 和 AppCode(应用编码)
        """
        params = {"limit": str(limit)}
        if controller:
            params["Controller"] = controller
        if app_code:
            params["AppCode"] = app_code
        result = self._invoke("getpdffilelist", **params)
        errormsg = result.get("errormsg", "")
        if errormsg:
            raise RuntimeError(f"getpdffilelist SQL错误: {errormsg}")
        raw = result.get("result", "[]")
        if isinstance(raw, str):
            items = json.loads(raw)
        elif isinstance(raw, list):
            items = raw
        else:
            items = []
        # 标准化 key 名
        normalized = []
        for item in items:
            if isinstance(item, dict):
                normalized.append({
                    "fileid": item.get("fileid", item.get("FileId", item.get("fileId", ""))),
                    "objectid": item.get("objectid", item.get("ObjectId", item.get("objectId", ""))),
                })
        return normalized

    def update_biz_object(self, schema_code: str, biz_object_id: str,
                          field_values: dict) -> dict:
        """更新氚云表单字段。BizObject 为 JSON 字符串格式（非嵌套对象）"""
        biz_object_str = json.dumps(field_values, ensure_ascii=False)
        return self._invoke("UpdateBizObject", SchemaCode=schema_code,
                            BizObjectId=biz_object_id, BizObject=biz_object_str)

    def download_attachment(self, file_id: str, obj_id: str,
                            save_dir: Path) -> Optional[Path]:
        """下载氚云附件（multipart/form-data，与 QuoteAI 一致）"""
        save_dir.mkdir(parents=True, exist_ok=True)
        boundary = "----H3YunDL" + format(int(time.time() * 1000), 'x')
        CRLF = "\r\n"
        parts = [
            f"--{boundary}{CRLF}Content-Disposition: form-data; name=\"attachmentId\"{CRLF}{CRLF}{file_id}{CRLF}",
            f"--{boundary}{CRLF}Content-Disposition: form-data; name=\"EngineCode\"{CRLF}{CRLF}{self.engine_code}{CRLF}",
            f"--{boundary}{CRLF}Content-Disposition: form-data; name=\"EngineSecret\"{CRLF}{CRLF}{self.engine_secret}{CRLF}",
            f"--{boundary}--{CRLF}",
        ]
        request_body = "".join(parts).encode("utf-8")
        url = f"{self.base_url}/Api/DownloadBizObjectFile"
        headers = {
            "EngineCode": self.engine_code,
            "EngineSecret": self.engine_secret,
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "Content-Length": str(len(request_body)),
        }
        try:
            resp = requests.post(url, headers=headers, data=request_body, timeout=120)
            if resp.status_code != 200:
                log.error("下载HTTP %d", resp.status_code)
                return None
            content = resp.content
            if len(content) < 100:
                log.error("下载响应过短(%d bytes)", len(content))
                return None
            cd = resp.headers.get("Content-Disposition", "")
            ext, filename = ".pdf", "attachment"
            if cd:
                match = re.search(r'filename[^;=\n]*=((["\']).*?\2|[^;\n]*)', cd)
                if match and match.group(1):
                    filename = match.group(1).strip('"').strip("'")
                    try:
                        from urllib.parse import unquote
                        filename = unquote(filename)
                    except Exception:
                        pass
                    fl = filename.lower()
                    if fl.endswith(".dwg"): ext = ".dwg"
                    elif fl.endswith((".jpg", ".png")): ext = ".png"
                    elif fl.endswith((".step", ".stp")): ext = ".step"
            local_path = save_dir / f"h3yun_{obj_id}_{int(time.time())}{ext}"
            local_path.write_bytes(content)
            log.info("下载OK (%s, %.0fKB)", filename, len(content) / 1024)
            return local_path
        except Exception as e:
            log.error("下载异常: %s", e)
            return None


# ────────────────────── Pdfsim 客户端 ──────────────────────
class PdfsimClient:
    def __init__(self, base_url: str = "http://127.0.0.1:8100"):
        self.base_url = base_url.rstrip("/")

    def health(self) -> dict:
        r = requests.get(f"{self.base_url}/api/v1/health", timeout=10)
        r.raise_for_status()
        return r.json()

    def upload(self, file_path: Path, allow_existing: bool = True,
               h3yun_object_id: str = "", h3yun_schema_code: str = "") -> dict:
        with open(file_path, "rb") as f:
            r = requests.post(
                f"{self.base_url}/api/v1/documents",
                files={"file": (file_path.name, f, "application/pdf")},
                data={"allow_existing": str(allow_existing).lower(),
                      "h3yun_object_id": h3yun_object_id,
                      "h3yun_schema_code": h3yun_schema_code},
                timeout=120)
        r.raise_for_status()
        return r.json()

    def record_import(self, h3yun_object_id: str, h3yun_schema_code: str = "",
                      doc_id: str = "", filename: str = "",
                      status: str = STATUS_PENDING, error_message: str = "") -> dict:
        r = requests.post(
            f"{self.base_url}/api/v1/import/record",
            params={"h3yun_object_id": h3yun_object_id,
                    "h3yun_schema_code": h3yun_schema_code,
                    "doc_id": doc_id, "filename": filename,
                    "status": status, "error_message": error_message},
            timeout=10)
        r.raise_for_status()
        return r.json()

    def get_import_status(self) -> dict:
        r = requests.get(f"{self.base_url}/api/v1/import/status", timeout=10)
        r.raise_for_status()
        return r.json()


# ────────────────────── 辅助函数 ──────────────────────

def _write_back(h3: H3YunClient, schema_code: str, obj_id: str,
                status_field: str, value: str):
    """回写氚云标识字段"""
    try:
        h3.update_biz_object(schema_code, obj_id, {status_field: value})
        log.info("回写氚云(%s=%s) OK", status_field, value[:30])
    except Exception as e:
        log.error("回写氚云失败(%s), 需手动标记!", e)


def _process_one(h3: H3YunClient, pdfsim: PdfsimClient,
                 schema_code: str, status_field: str,
                 download_dir: Path,
                 obj_id: str, file_id: str, name: str,
                 idx: int, total: int) -> str:
    """处理单条: 下载 → 验证PDF → 入库 → 回写。返回最终状态"""
    log.info("[%d/%d] %s (fileid=%s...)", idx, total, name, file_id[:12])
    pdfsim.record_import(h3yun_object_id=obj_id, h3yun_schema_code=schema_code,
                         filename=name, status=STATUS_INGESTING)

    # ── 下载 ──
    try:
        pdf_path = h3.download_attachment(file_id, obj_id, download_dir)
        if pdf_path is None:
            resp = pdfsim.record_import(h3yun_object_id=obj_id, h3yun_schema_code=schema_code,
                                       filename=name, status=STATUS_DOWNLOAD_FAIL,
                                       error_message="附件下载失败")
            if resp.get("status") == STATUS_DEAD:
                _write_back(h3, schema_code, obj_id, status_field, "入库失败")
            return STATUS_DOWNLOAD_FAIL
    except Exception as e:
        log.error("下载异常: %s", e)
        resp = pdfsim.record_import(h3yun_object_id=obj_id, h3yun_schema_code=schema_code,
                                   filename=name, status=STATUS_DOWNLOAD_FAIL,
                                   error_message=f"下载异常: {e}")
        if resp.get("status") == STATUS_DEAD:
            _write_back(h3, schema_code, obj_id, status_field, "入库失败")
        return STATUS_DOWNLOAD_FAIL

    # ── 验证PDF ──
    content = pdf_path.read_bytes()
    pdf_path.unlink(missing_ok=True)

    if len(content) < 10:
        log.warning("空文件")
        pdfsim.record_import(h3yun_object_id=obj_id, h3yun_schema_code=schema_code,
                             filename=name, status=STATUS_NON_PDF, error_message="空文件")
        _write_back(h3, schema_code, obj_id, status_field, "非PDF")
        return STATUS_NON_PDF

    if content[:5] != b"%PDF-":
        ext = Path(pdf_path.name).suffix if pdf_path else ""
        log.warning("非PDF(%s)", ext)
        pdfsim.record_import(h3yun_object_id=obj_id, h3yun_schema_code=schema_code,
                             filename=name, status=STATUS_NON_PDF, error_message=f"非PDF({ext})")
        _write_back(h3, schema_code, obj_id, status_field, "非PDF")
        return STATUS_NON_PDF

    # ── 入库 Pdfsim ──
    tmp_path = download_dir / f"h3yun_{obj_id}_{int(time.time())}.pdf"
    tmp_path.write_bytes(content)
    doc_id = ""
    try:
        result = pdfsim.upload(tmp_path, allow_existing=True,
                               h3yun_object_id=obj_id, h3yun_schema_code=schema_code)
        if result.get("ok"):
            doc_id = result.get("doc_id", "")
            log.info("入库OK (doc_id=%s, %d页)", doc_id, result.get("num_pages", 0))
        else:
            msg = result.get("message", "unknown")
            log.error("入库失败: %s", msg)
            resp = pdfsim.record_import(h3yun_object_id=obj_id, h3yun_schema_code=schema_code,
                                       filename=name, status=STATUS_FAILED, error_message=msg[:500])
            if resp.get("status") == STATUS_DEAD:
                _write_back(h3, schema_code, obj_id, status_field, "入库失败")
            tmp_path.unlink(missing_ok=True)
            return STATUS_FAILED
    except requests.exceptions.HTTPError as e:
        detail = str(e)
        try: detail = e.response.json().get("detail", str(e))
        except Exception: pass
        log.error("入库HTTP错误: %s", detail)
        st = STATUS_PARSE_FAIL if ("解析失败" in detail or "PDF" in detail) else STATUS_FAILED
        resp = pdfsim.record_import(h3yun_object_id=obj_id, h3yun_schema_code=schema_code,
                                   filename=name, status=st, error_message=detail[:500])
        if st == STATUS_PARSE_FAIL:
            _write_back(h3, schema_code, obj_id, status_field, "PDF损坏")
        elif resp.get("status") == STATUS_DEAD:
            _write_back(h3, schema_code, obj_id, status_field, "入库失败")
        tmp_path.unlink(missing_ok=True)
        return st
    except Exception as e:
        log.error("入库异常: %s", e)
        resp = pdfsim.record_import(h3yun_object_id=obj_id, h3yun_schema_code=schema_code,
                                   filename=name, status=STATUS_FAILED,
                                   error_message=f"入库异常: {e}"[:500])
        if resp.get("status") == STATUS_DEAD:
            _write_back(h3, schema_code, obj_id, status_field, "入库失败")
        tmp_path.unlink(missing_ok=True)
        return STATUS_FAILED

    tmp_path.unlink(missing_ok=True)

    # ★ 成功 → 回写 doc_id
    _write_back(h3, schema_code, obj_id, status_field, doc_id or "已入库")
    pdfsim.record_import(h3yun_object_id=obj_id, h3yun_schema_code=schema_code,
                         doc_id=doc_id, filename=name, status=STATUS_DONE)
    return STATUS_DONE


# ────────────────────── scan 阶段 ──────────────────────

def phase_scan(config: dict, limit: int = 10):
    h3 = H3YunClient(engine_code=config["h3yun"]["engine_code"],
                     engine_secret=config["h3yun"]["engine_secret"],
                     base_url=config["h3yun"].get("base_url", "https://www.h3yun.com"))
    pdfsim = PdfsimClient(config["pdfsim_url"])
    schema_code = config["h3yun"]["product_schema_code"]

    try:
        health = pdfsim.health()
        log.info("Pdfsim: %d 文档 / %d 向量", health.get("docs", 0), health.get("vectors", 0))
    except Exception as e:
        log.error("Pdfsim 不可达: %s", e); return

    log.info("调用 getpdffilelist(limit=%d)...", limit)
    try:
        items = h3.get_pdf_file_list(limit=limit,
                                     controller=config.get("h3yun_controller", ""),
                                     app_code=config.get("h3yun_app_code", ""))
    except Exception as e:
        log.error("getpdffilelist 失败: %s", e); return

    if not items:
        log.info("没有待处理的PDF图纸"); return

    log.info("找到 %d 个待处理PDF:", len(items))
    for i, item in enumerate(items):
        log.info("  [%d] objectid=%s  fileid=%s", i + 1, item["objectid"], item["fileid"])

    # 记录到 Pdfsim import_status
    for item in items:
        pdfsim.record_import(
            h3yun_object_id=item["objectid"],
            h3yun_schema_code=schema_code,
            filename=f"obj_{item['objectid'][:8]}",
            status=STATUS_PENDING,
            error_message=f"fileid={item['fileid']}"
        )

    log.info("扫描完成！%d 个待入库", len(items))


# ────────────────────── import 阶段（单批） ──────────────────────

def phase_import(config: dict, limit: int = 10):
    h3 = H3YunClient(engine_code=config["h3yun"]["engine_code"],
                     engine_secret=config["h3yun"]["engine_secret"],
                     base_url=config["h3yun"].get("base_url", "https://www.h3yun.com"))
    pdfsim = PdfsimClient(config["pdfsim_url"])
    schema_code = config["h3yun"]["product_schema_code"]
    status_field = config.get("status_field_code", "F0000223")
    download_dir = Path(config.get("download_dir", "/tmp/pdfsim_downloads"))
    download_dir.mkdir(parents=True, exist_ok=True)

    try:
        health = pdfsim.health()
        log.info("Pdfsim: %d 文档 / %d 向量", health.get("docs", 0), health.get("vectors", 0))
    except Exception as e:
        log.error("Pdfsim 不可达: %s", e); return

    log.info("调用 getpdffilelist(limit=%d)...", limit)
    try:
        items = h3.get_pdf_file_list(limit=limit,
                                     controller=config.get("h3yun_controller", ""),
                                     app_code=config.get("h3yun_app_code", ""))
    except Exception as e:
        log.error("getpdffilelist 失败: %s", e); return

    if not items:
        log.info("没有待处理的PDF图纸"); return

    total = len(items)
    log.info("待入库: %d 个", total)

    success = failed = skipped = 0
    for i, item in enumerate(items):
        obj_id = item["objectid"]
        file_id = item["fileid"]
        name = f"obj_{obj_id[:8]}"

        result_status = _process_one(h3, pdfsim, schema_code, status_field,
                                     download_dir, obj_id, file_id, name,
                                     i + 1, total)

        if result_status == STATUS_DONE:
            success += 1
        elif result_status in (STATUS_NON_PDF,):
            skipped += 1
        else:
            failed += 1
        time.sleep(0.5)

    log.info("入库完成！成功=%d 失败=%d 跳过=%d", success, failed, skipped)


# ────────────────────── daemon 模式：持续轮询入库 ──────────────────────

def phase_daemon(config: dict, limit: int = 10, interval: int = 300):
    """
    持续循环入库（服务模式）：
    1. 调 getpdffilelist(limit) 拿一批
    2. 逐条处理（下载→入库→回写F0000223）
    3. 处理完 sleep interval 秒
    4. 继续下一轮——即使当前没数据也继续轮询（氚云每天都有新增）
    5. Ctrl+C 优雅停止
    """
    h3 = H3YunClient(engine_code=config["h3yun"]["engine_code"],
                     engine_secret=config["h3yun"]["engine_secret"],
                     base_url=config["h3yun"].get("base_url", "https://www.h3yun.com"))
    pdfsim = PdfsimClient(config["pdfsim_url"])
    schema_code = config["h3yun"]["product_schema_code"]
    status_field = config.get("status_field_code", "F0000223")
    download_dir = Path(config.get("download_dir", "/tmp/pdfsim_downloads"))
    download_dir.mkdir(parents=True, exist_ok=True)

    log.info("=" * 50)
    log.info("daemon 模式启动！每批 %d 条，轮询间隔 %ds", limit, interval)
    log.info("氚云新产品会自动被拉取入库，Ctrl+C 停止")
    log.info("=" * 50)

    round_num = 0
    total_success = total_failed = total_skipped = 0
    consecutive_empty = 0

    while True:
        round_num += 1

        # 检查 Pdfsim
        try:
            health = pdfsim.health()
            docs, vectors = health.get("docs", 0), health.get("vectors", 0)
        except Exception as e:
            log.warning("Pdfsim 不可达(%s), %ds 后重试", e, interval)
            time.sleep(interval)
            continue

        # 拉待处理
        try:
            items = h3.get_pdf_file_list(limit=limit,
                                     controller=config.get("h3yun_controller", ""),
                                     app_code=config.get("h3yun_app_code", ""))
        except Exception as e:
            log.warning("getpdffilelist 失败(%s), %ds 后重试", e, interval)
            time.sleep(interval)
            continue

        if not items:
            consecutive_empty += 1
            # 没数据也继续轮询，因为氚云每天都有新增
            log.info("第%d轮: 无待处理 | 库中 %d 文档/%d 向量 | %ds 后下一轮...",
                     round_num, docs, vectors, interval)
            time.sleep(interval)
            continue

        consecutive_empty = 0
        batch_success = batch_failed = batch_skipped = 0
        log.info("第%d轮: 拿到 %d 条 | 库中 %d 文档/%d 向量", round_num, len(items), docs, vectors)

        for i, item in enumerate(items):
            obj_id = item["objectid"]
            file_id = item["fileid"]
            name = f"obj_{obj_id[:8]}"

            try:
                result_status = _process_one(h3, pdfsim, schema_code, status_field,
                                             download_dir, obj_id, file_id, name,
                                             i + 1, len(items))
            except Exception as e:
                log.error("处理异常: %s", e)
                result_status = STATUS_FAILED

            if result_status == STATUS_DONE:
                batch_success += 1
            elif result_status in (STATUS_NON_PDF,):
                batch_skipped += 1
            else:
                batch_failed += 1

            time.sleep(0.5)

        total_success += batch_success
        total_failed += batch_failed
        total_skipped += batch_skipped

        log.info("本轮: 成功=%d 失败=%d 跳过=%d", batch_success, batch_failed, batch_skipped)
        log.info("累计: 成功=%d 失败=%d 跳过=%d | %ds 后下一轮...", 
                 total_success, total_failed, total_skipped, interval)
        time.sleep(interval)


# ────────────────────── report ──────────────────────

def phase_report(config: dict):
    pdfsim = PdfsimClient(config["pdfsim_url"])
    try:
        stats = pdfsim.get_import_status()
    except Exception as e:
        log.error("Pdfsim 不可达: %s", e); return

    total = stats.get("total", 0)
    by_status = stats.get("by_status", {})
    print(f"\n{'='*50}\n图纸入库状态报告\n{'='*50}")
    print(f"  总计: {total}\n")
    for status in [STATUS_DONE, STATUS_PENDING, STATUS_INGESTING,
                   STATUS_NON_PDF, STATUS_PARSE_FAIL,
                   STATUS_DOWNLOAD_FAIL, STATUS_FAILED, STATUS_DEAD]:
        c = by_status.get(status, 0)
        if c > 0:
            print(f"  {status:<16} {c:>6}  {STATUS_LABELS.get(status, status)}")

    actionable = by_status.get(STATUS_PENDING, 0)
    retryable = sum(by_status.get(s, 0) for s in RETRYABLE_STATUSES)
    print(f"\n  --- 行动项 ---")
    if actionable: print(f"  ▸ {actionable} 待入库: python scripts/h3yun_batch_import.py import")
    if retryable: print(f"  ▸ {retryable} 可重试: python scripts/h3yun_batch_import.py import")


# ────────────────────── 配置加载 ──────────────────────

def _load_config(config_path: str) -> dict:
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from app.config import settings
        h3 = settings.h3yun
        if h3.engine_code and h3.product_schema_code:
            return {
                "pdfsim_url": "http://127.0.0.1:8100",
                "h3yun": {"engine_code": h3.engine_code, "engine_secret": h3.engine_secret,
                          "base_url": h3.base_url, "product_schema_code": h3.product_schema_code},
                "status_field_code": getattr(h3, "name_field_code", "F0000223"),
                "download_dir": "/tmp/pdfsim_downloads",
                # 自定义action必需参数
                "h3yun_controller": getattr(h3, "controller", ""),
                "h3yun_app_code": getattr(h3, "app_code", ""),
            }
    except Exception: pass
    cp = Path(config_path)
    if cp.exists():
        with open(cp) as f: return json.load(f)
    log.error("没有可用配置！"); sys.exit(1)


# ────────────────────── 主入口 ──────────────────────

def main():
    parser = argparse.ArgumentParser(description="氚云图纸批量入库 v4")
    parser.add_argument("action", choices=["scan", "import", "daemon", "report"])
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--limit", type=int, default=10, help="每批数量(默认10)")
    parser.add_argument("--interval", type=int, default=300, help="daemon轮询间隔秒数(默认300)")
    args = parser.parse_args()
    config = _load_config(args.config)

    if args.action == "scan":
        phase_scan(config, limit=args.limit)
    elif args.action == "import":
        phase_import(config, limit=args.limit)
    elif args.action == "daemon":
        try:
            phase_daemon(config, limit=args.limit, interval=args.interval)
        except KeyboardInterrupt:
            log.info("daemon 已停止")
    elif args.action == "report":
        phase_report(config)


if __name__ == "__main__":
    main()
