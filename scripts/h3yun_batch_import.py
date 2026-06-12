"""
氚云图纸批量入库脚本 v3

核心逻辑:
  1. LoadBizObjects + Filter(F0000223为空) 只拉未处理的
  2. 附件字段是数组，取最后一个(最新的)
  3. DownloadBizObjectFile 下载附件(multipart/form-data)
  4. 处理完一律回写 F0000223，确保不再被扫描到:
     - 成功 → doc_id
     - 失败3次(dead) → "入库失败"
     - 非PDF → "非PDF"
  5. 重试上限3次：后端自动管理 retry_count，>=3 标记 dead

使用方式：
  python scripts/h3yun_batch_import.py scan   # 扫描
  python scripts/h3yun_batch_import.py import # 入库+回写
  python scripts/h3yun_batch_import.py report # 报告
  python scripts/h3yun_batch_import.py retry  # 重试失败(≤3次)
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import List, Optional

import requests

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

    def load_biz_objects(self, schema_code: str, filter_str: str = "",
                         from_row: int = 0, to_row: int = 500) -> List[dict]:
        """批量加载表单数据。Filter 是氚云专用 JSON 格式字符串。"""
        result = self._invoke("LoadBizObjects", SchemaCode=schema_code,
                              Filter=filter_str)
        if isinstance(result, list):
            return result
        if isinstance(result, dict):
            # 氚云返回 key 是 BizObjectArray
            return result.get("BizObjectArray", result.get("BizObjects", result.get("Data", [])))
        return []

    @staticmethod
    def build_filter(status_field: str = "", status_value: str = "",
                     from_row: int = 0, to_row: int = 500) -> str:
        """
        构建氚云 LoadBizObjects 的 Filter 参数。
        氚云 Filter 格式：
        {
          "FromRowNum": 0, "ToRowNum": 500,
          "RequireCount": false, "ReturnItems": [], "SortByCollection": [],
          "Matcher": {"Type": "And", "Matchers": [
            {"Type": "Item", "Name": "字段编码", "Operator": 2, "Value": "值"}
          ]}
        }
        Operator: 0=大于, 1=大于等于, 2=等于, 3=小于等于, 4=小于, 5=不等于
        """
        matchers = []
        if status_field:
            matchers.append({"Type": "Item", "Name": status_field,
                             "Operator": 2, "Value": status_value})
        filter_obj = {
            "FromRowNum": from_row,
            "ToRowNum": to_row,
            "RequireCount": False,
            "ReturnItems": [],
            "SortByCollection": [],
            "Matcher": {"Type": "And", "Matchers": matchers}
        }
        return json.dumps(filter_obj)

    def update_biz_object(self, schema_code: str, biz_object_id: str,
                          field_values: dict) -> dict:
        return self._invoke("UpdateBizObject", SchemaCode=schema_code,
                            BizObjectId=biz_object_id, **field_values)

    @staticmethod
    def parse_attachment_field(value) -> List[str]:
        if not value:
            return []
        if isinstance(value, list):
            ids = []
            for item in value:
                if isinstance(item, str):
                    ids.append(item)
                elif isinstance(item, dict):
                    ids.append(item.get("Id") or item.get("id") or item.get("FileId") or "")
            return [x for x in ids if x]
        if not isinstance(value, str):
            return []
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                ids = []
                for item in parsed:
                    if isinstance(item, str):
                        ids.append(item)
                    elif isinstance(item, dict):
                        ids.append(item.get("Id") or item.get("id") or item.get("FileId") or "")
                return [x for x in ids if x]
            if isinstance(parsed, dict):
                fid = parsed.get("Id") or parsed.get("id")
                return [fid] if fid else []
        except (json.JSONDecodeError, ValueError):
            pass
        if ":" in value:
            return [p.split(":")[0].strip() for p in value.split(";") if ":" in p and p.strip()]
        if value.strip():
            return [value.strip()]
        return []

    def download_attachment(self, file_id: str, obj_id: str,
                            save_dir: Path) -> Optional[Path]:
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
                print(f"  [error] HTTP {resp.status_code}")
                return None
            content = resp.content
            if len(content) < 100:
                print(f"  [error] 响应异常(过短)")
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
            print(f"OK ({filename}, {len(content)/1024:.0f}KB)")
            return local_path
        except Exception as e:
            print(f"  [error] {e}")
            return None


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

    def get_import_failed(self, limit: int = 200) -> dict:
        r = requests.get(f"{self.base_url}/api/v1/import/failed", params={"limit": limit}, timeout=10)
        r.raise_for_status()
        return r.json()

    def get_import_pending(self, limit: int = 200) -> dict:
        r = requests.get(f"{self.base_url}/api/v1/import/pending", params={"limit": limit}, timeout=10)
        r.raise_for_status()
        return r.json()


def _write_back(h3, schema_code, obj_id, status_field, value):
    """回写氚云标识字段，确保已处理项不再被扫描"""
    try:
        h3.update_biz_object(schema_code, obj_id, {status_field: value})
        print(f"  回写氚云({status_field}={value}) OK")
    except Exception as e:
        print(f"  回写氚云失败({e}), 需手动标记!")


def phase_scan(config: dict, limit: int = 0):
    h3 = H3YunClient(engine_code=config["h3yun"]["engine_code"],
                     engine_secret=config["h3yun"]["engine_secret"],
                     base_url=config["h3yun"].get("base_url", "https://www.h3yun.com"))
    pdfsim = PdfsimClient(config["pdfsim_url"])
    schema_code = config["h3yun"]["product_schema_code"]
    attachment_field = config.get("attachment_field_code", "F0000052")
    status_field = config.get("status_field_code", "F0000223")

    try:
        health = pdfsim.health()
        print(f"[info] Pdfsim: {health.get('docs',0)} 文档 / {health.get('vectors',0)} 向量")
    except Exception as e:
        print(f"[error] Pdfsim 不可达: {e}"); return

    print(f"\n[scan] 拉取未处理产品 (Filter: {status_field}为空)...")
    all_objects = _load_all_pages(h3, schema_code, status_field, "")
    if not all_objects:
        print("[info] 没有未处理的产品"); return
    if limit > 0:
        all_objects = all_objects[:limit]

    total = len(all_objects)
    has_att = no_att = 0
    for i, obj in enumerate(all_objects):
        obj_id = obj.get("ObjectId", "")
        name = obj.get("Name", obj_id)
        if not obj_id: continue
        file_ids = h3.parse_attachment_field(obj.get(attachment_field, ""))
        if not file_ids:
            # 无附件也回写，避免重复扫描
            _write_back(h3, schema_code, obj_id, status_field, "无附件")
            pdfsim.record_import(h3yun_object_id=obj_id, h3yun_schema_code=schema_code,
                                 filename=name, status=STATUS_NO_ATTACHMENT)
            no_att += 1; continue
        has_att += 1
        att_info = f"{len(file_ids)}个附件, 取最新"
        print(f"  [{i+1}/{total}] {name}: {att_info}")
        pdfsim.record_import(h3yun_object_id=obj_id, h3yun_schema_code=schema_code,
                             filename=name, status=STATUS_PENDING, attachment_info=att_info)
    print(f"\n扫描完成！未处理: {total}, 有附件: {has_att}, 无附件(已回写): {no_att}")


def phase_import(config: dict, limit: int = 0, retry: bool = False):
    h3 = H3YunClient(engine_code=config["h3yun"]["engine_code"],
                     engine_secret=config["h3yun"]["engine_secret"],
                     base_url=config["h3yun"].get("base_url", "https://www.h3yun.com"))
    pdfsim = PdfsimClient(config["pdfsim_url"])
    schema_code = config["h3yun"]["product_schema_code"]
    attachment_field = config.get("attachment_field_code", "F0000052")
    status_field = config.get("status_field_code", "F0000223")
    download_dir = Path(config.get("download_dir", "/tmp/pdfsim_downloads"))
    download_dir.mkdir(parents=True, exist_ok=True)

    try:
        health = pdfsim.health()
        print(f"[info] Pdfsim: {health.get('docs',0)} 文档 / {health.get('vectors',0)} 向量")
    except Exception as e:
        print(f"[error] Pdfsim 不可达: {e}"); return

    print(f"\n[import] 拉取未处理产品...")
    all_objects = _load_all_pages(h3, schema_code, status_field, "")

    # ObjectId -> 附件映射
    obj_map = {}
    for obj in all_objects:
        obj_id = obj.get("ObjectId", "")
        name = obj.get("Name", obj_id)
        file_ids = h3.parse_attachment_field(obj.get(attachment_field, ""))
        obj_map[obj_id] = {"name": name, "file_id": file_ids[-1] if file_ids else "",
                           "total_attachments": len(file_ids)}

    # 从 PG 获取待处理项
    pending_items = []
    try:
        pending_items = pdfsim.get_import_pending(limit=10000).get("items", [])
    except Exception: pass
    if retry:
        try:
            pending_items += pdfsim.get_import_failed(limit=10000).get("items", [])
        except Exception: pass

    if not pending_items:
        print("[info] 没有待处理的项目"); return

    print(f"[info] 待入库: {len(pending_items)} 个")
    if limit > 0:
        pending_items = pending_items[:limit]

    success = failed = skipped = dead_count = 0

    for i, item in enumerate(pending_items):
        obj_id = item.get("h3yun_object_id", "")
        name = item.get("filename", obj_id)
        obj_info = obj_map.get(obj_id, {})
        file_id = obj_info.get("file_id", "")

        if not file_id:
            _write_back(h3, schema_code, obj_id, status_field, "无附件")
            pdfsim.record_import(h3yun_object_id=obj_id, h3yun_schema_code=schema_code,
                                 filename=name, status=STATUS_NO_ATTACHMENT,
                                 error_message="二次确认无附件")
            skipped += 1; continue

        total_att = obj_info.get("total_attachments", 1)
        print(f"\n[{i+1}/{len(pending_items)}] {name} ({total_att}个附件取最新)")
        pdfsim.record_import(h3yun_object_id=obj_id, h3yun_schema_code=schema_code,
                             filename=name, status=STATUS_INGESTING)

        # 下载
        print(f"  下载 {file_id}...", end=" ", flush=True)
        pdf_path = None
        try:
            pdf_path = h3.download_attachment(file_id, obj_id, download_dir)
            if pdf_path is None:
                resp = pdfsim.record_import(h3yun_object_id=obj_id, h3yun_schema_code=schema_code,
                                           filename=name, status=STATUS_DOWNLOAD_FAIL,
                                           error_message="附件下载失败")
                if resp.get("status") == STATUS_DEAD:
                    _write_back(h3, schema_code, obj_id, status_field, "入库失败")
                    dead_count += 1
                failed += 1; continue
        except Exception as e:
            print(f"异常({e})")
            resp = pdfsim.record_import(h3yun_object_id=obj_id, h3yun_schema_code=schema_code,
                                       filename=name, status=STATUS_DOWNLOAD_FAIL,
                                       error_message=f"下载异常: {e}")
            if resp.get("status") == STATUS_DEAD:
                _write_back(h3, schema_code, obj_id, status_field, "入库失败")
                dead_count += 1
            failed += 1; continue

        # 检查PDF
        content = pdf_path.read_bytes()
        pdf_path.unlink(missing_ok=True)

        if len(content) < 10:
            print("  空文件")
            pdfsim.record_import(h3yun_object_id=obj_id, h3yun_schema_code=schema_code,
                                 filename=name, status=STATUS_NON_PDF, error_message="空文件")
            _write_back(h3, schema_code, obj_id, status_field, "非PDF")
            skipped += 1; continue
        if content[:5] != b"%PDF-":
            ext = Path(pdf_path.name).suffix if pdf_path else ""
            print(f"  非PDF({ext})")
            pdfsim.record_import(h3yun_object_id=obj_id, h3yun_schema_code=schema_code,
                                 filename=name, status=STATUS_NON_PDF, error_message=f"非PDF({ext})")
            _write_back(h3, schema_code, obj_id, status_field, "非PDF")
            skipped += 1; continue

        # 入库
        print(f"  入库...", end=" ", flush=True)
        doc_id = ""
        # 重新写到临时文件供 upload
        tmp_path = download_dir / f"h3yun_{obj_id}_{int(time.time())}.pdf"
        tmp_path.write_bytes(content)
        try:
            result = pdfsim.upload(tmp_path, allow_existing=True,
                                   h3yun_object_id=obj_id, h3yun_schema_code=schema_code)
            if result.get("ok"):
                doc_id = result.get("doc_id", "")
                print(f"OK (doc_id={doc_id}, {result.get('num_pages',0)}页)")
            else:
                msg = result.get("message", "unknown")
                print(f"失败({msg})")
                resp = pdfsim.record_import(h3yun_object_id=obj_id, h3yun_schema_code=schema_code,
                                           filename=name, status=STATUS_FAILED, error_message=msg[:500])
                if resp.get("status") == STATUS_DEAD:
                    _write_back(h3, schema_code, obj_id, status_field, "入库失败")
                    dead_count += 1
                failed += 1; tmp_path.unlink(missing_ok=True); continue
        except requests.exceptions.HTTPError as e:
            detail = str(e)
            try: detail = e.response.json().get("detail", str(e))
            except Exception: pass
            print(f"失败({detail})")
            st = STATUS_PARSE_FAIL if ("解析失败" in detail or "PDF" in detail) else STATUS_FAILED
            resp = pdfsim.record_import(h3yun_object_id=obj_id, h3yun_schema_code=schema_code,
                                       filename=name, status=st, error_message=detail[:500])
            # 终态（PDF损坏）也回写
            if st == STATUS_PARSE_FAIL:
                _write_back(h3, schema_code, obj_id, status_field, "PDF损坏")
            elif resp.get("status") == STATUS_DEAD:
                _write_back(h3, schema_code, obj_id, status_field, "入库失败")
                dead_count += 1
            failed += 1; tmp_path.unlink(missing_ok=True); continue
        except Exception as e:
            print(f"异常({e})")
            resp = pdfsim.record_import(h3yun_object_id=obj_id, h3yun_schema_code=schema_code,
                                       filename=name, status=STATUS_FAILED,
                                       error_message=f"入库异常: {e}"[:500])
            if resp.get("status") == STATUS_DEAD:
                _write_back(h3, schema_code, obj_id, status_field, "入库失败")
                dead_count += 1
            failed += 1; tmp_path.unlink(missing_ok=True); continue

        tmp_path.unlink(missing_ok=True)

        # ★ 成功 → 回写 doc_id
        _write_back(h3, schema_code, obj_id, status_field, doc_id or "已入库")
        pdfsim.record_import(h3yun_object_id=obj_id, h3yun_schema_code=schema_code,
                             doc_id=doc_id, filename=name, status=STATUS_DONE)
        success += 1
        print(f"  ✓ 完成")
        time.sleep(1)

    print(f"\n{'='*50}")
    print(f"入库完成！成功: {success}, 失败: {failed}, 跳过: {skipped}, dead(已回写): {dead_count}")


def _load_all_pages(h3, schema_code, status_field="", status_value="") -> List[dict]:
    """分页加载所有数据。如果 status_field 为空则不过滤，拉全量后脚本端过滤。"""
    all_objects = []
    from_row = 0
    page_size = 500  # 氚云最大500
    while True:
        # 氚云 Filter 空值匹配不可靠，先不加 Matcher 拉全量
        filter_obj = {
            "FromRowNum": from_row,
            "ToRowNum": from_row + page_size,
            "RequireCount": False,
            "ReturnItems": [],
            "SortByCollection": [],
            "Matcher": {"Type": "And", "Matchers": []}
        }
        filter_str = json.dumps(filter_obj)
        try:
            batch = h3.load_biz_objects(schema_code, filter_str=filter_str)
        except Exception as e:
            print(f"[error] from_row={from_row} 失败: {e}"); break
        if not batch: break
        all_objects.extend(batch)
        print(f"  from_row={from_row}: {len(batch)} 条")
        if len(batch) < page_size: break
        from_row += page_size
        time.sleep(0.5)
    # 脚本端过滤：只保留 status_field 为空或 None 的
    if status_field:
        filtered = []
        for obj in all_objects:
            val = obj.get(status_field)
            if val is None or str(val).strip() == "":
                filtered.append(obj)
        print(f"  全量 {len(all_objects)} 条, 过滤后({status_field}为空) {len(filtered)} 条")
        return filtered
    return all_objects


def phase_report(config: dict):
    pdfsim = PdfsimClient(config["pdfsim_url"])
    try:
        stats = pdfsim.get_import_status()
    except Exception as e:
        print(f"[error] {e}"); return

    total = stats.get("total", 0)
    by_status = stats.get("by_status", {})
    print(f"\n{'='*50}\n图纸入库状态报告\n{'='*50}")
    print(f"  总计: {total}\n")
    for status in [STATUS_DONE, STATUS_PENDING, STATUS_INGESTING,
                   STATUS_NO_ATTACHMENT, STATUS_NON_PDF, STATUS_PARSE_FAIL,
                   STATUS_DOWNLOAD_FAIL, STATUS_FAILED, STATUS_DEAD]:
        c = by_status.get(status, 0)
        if c > 0:
            print(f"  {status:<16} {c:>6}  {STATUS_LABELS.get(status, status)}")

    actionable = by_status.get(STATUS_PENDING, 0)
    need_human = sum(by_status.get(s, 0) for s in TERMINAL_STATUSES)
    retryable = sum(by_status.get(s, 0) for s in RETRYABLE_STATUSES)
    print(f"\n  --- 行动项 ---")
    if actionable: print(f"  ▸ {actionable} 待入库: python scripts/h3yun_batch_import.py import")
    if retryable: print(f"  ▸ {retryable} 可重试: python scripts/h3yun_batch_import.py retry")
    if need_human:
        print(f"  ▸ {need_human} 需人工:")
        for s in TERMINAL_STATUSES:
            c = by_status.get(s, 0)
            if c: print(f"    - {STATUS_LABELS[s]}: {c}")


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
                "attachment_field_code": h3.attachment_field_code,
                "status_field_code": h3.name_field_code,
                "download_dir": "/tmp/pdfsim_downloads",
            }
    except Exception: pass
    cp = Path(config_path)
    if cp.exists():
        with open(cp) as f: return json.load(f)
    print("[error] 没有可用配置！"); sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="氚云图纸批量入库")
    parser.add_argument("action", choices=["scan", "import", "report", "retry"])
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    config = _load_config(args.config)
    if args.action == "scan": phase_scan(config, limit=args.limit)
    elif args.action == "import": phase_import(config, limit=args.limit, retry=False)
    elif args.action == "retry": phase_import(config, limit=args.limit, retry=True)
    elif args.action == "report": phase_report(config)


if __name__ == "__main__":
    main()
