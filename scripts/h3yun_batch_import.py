"""
氚云图纸批量入库脚本 v2

核心思路：全量扫描，每种情况都打标签，不遗漏

状态体系：
  pending       - 刚发现，还未处理
  no_attachment - 无附件（需去氚云补）
  non_pdf       - 附件非PDF格式
  download_fail - 附件下载失败
  parse_fail    - PDF解析失败（损坏等）
  ingesting     - 正在入库
  done          - 已入库
  failed        - 入库失败（可重试）
  duplicate     - 重复图纸（已存在）

流程：
  Phase 1: 全量扫描 → 从氚云拉所有产品，逐个标记状态（不碰氚云原数据）
  Phase 2: 只处理状态为 pending 且有附件的
  Phase 3: 汇总报告

使用方式：
  python scripts/h3yun_batch_import.py scan   --config config.json   # 全量扫描
  python scripts/h3yun_batch_import.py import --config config.json   # 执行入库
  python scripts/h3yun_batch_import.py report --config config.json   # 查看报告
  python scripts/h3yun_batch_import.py retry  --config config.json   # 重试失败的
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import requests


# =====================
# 状态定义
# =====================

STATUS_PENDING = "pending"           # 刚发现，有附件，待入库
STATUS_NO_ATTACHMENT = "no_attachment"  # 无附件（需去氚云补）
STATUS_NON_PDF = "non_pdf"           # 附件非PDF格式
STATUS_DOWNLOAD_FAIL = "download_fail"  # 附件下载失败
STATUS_PARSE_FAIL = "parse_fail"     # PDF解析失败（损坏等）
STATUS_INGESTING = "ingesting"       # 正在入库
STATUS_DONE = "done"                 # 已入库
STATUS_FAILED = "failed"             # 入库失败（可重试）
STATUS_DUPLICATE = "duplicate"       # 重复图纸（已存在）

# 不可重试的终态（需要人工干预）
TERMINAL_STATUSES = {STATUS_NO_ATTACHMENT, STATUS_NON_PDF, STATUS_PARSE_FAIL}
# 可重试的状态
RETRYABLE_STATUSES = {STATUS_PENDING, STATUS_DOWNLOAD_FAIL, STATUS_FAILED}

STATUS_LABELS = {
    STATUS_PENDING: "待入库",
    STATUS_NO_ATTACHMENT: "无附件",
    STATUS_NON_PDF: "非PDF附件",
    STATUS_DOWNLOAD_FAIL: "下载失败",
    STATUS_PARSE_FAIL: "PDF损坏",
    STATUS_INGESTING: "入库中",
    STATUS_DONE: "已入库",
    STATUS_FAILED: "入库失败",
    STATUS_DUPLICATE: "重复图纸",
}


# =====================
# 氚云 API 封装（只读，不写原数据）
# =====================

class H3YunClient:
    """氚云 OpenAPI 简易客户端（只读）"""

    def __init__(self, engine_code: str, engine_secret: str,
                 base_url: str = "https://www.h3yun.com"):
        self.engine_code = engine_code
        self.engine_secret = engine_secret
        self.base_url = base_url
        self.api_url = f"{base_url}/OpenApi/Invoke"

    def _invoke(self, action: str, payload: dict) -> dict:
        headers = {
            "EngineCode": self.engine_code,
            "EngineSecret": self.engine_secret,
            "Content-Type": "application/json",
        }
        body = {
            "ActionName": action,
            "Payload": json.dumps(payload),
        }
        resp = requests.post(self.api_url, json=body, headers=headers, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        if data.get("ReturnCode") != 0:
            raise RuntimeError(f"氚云API错误: {data.get('ReturnMsg', data)}")
        return data.get("ReturnData", {})

    def load_biz_objects(self, schema_code: str, page_size: int = 200,
                         page_index: int = 0) -> List[dict]:
        payload = {
            "SchemaCode": schema_code,
            "Filter": "",
            "IsGetSummary": False,
            "PageIndex": page_index,
            "PageSize": page_size,
        }
        result = self._invoke("LoadBizObjects", payload)
        return result if isinstance(result, list) else []

    def download_file(self, file_id: str, save_path: Path) -> bool:
        payload = {"FileId": file_id}
        try:
            headers = {
                "EngineCode": self.engine_code,
                "EngineSecret": self.engine_secret,
            }
            body = {
                "ActionName": "DownloadBizObjectFile",
                "Payload": json.dumps(payload),
            }
            resp = requests.post(self.api_url, json=body, headers=headers,
                                 timeout=120, stream=True)
            if resp.status_code == 200 and len(resp.content) > 0:
                save_path.write_bytes(resp.content)
                return True
            return False
        except Exception as e:
            print(f"  [error] 下载文件失败: {e}")
            return False


# =====================
# Pdfsim API 封装
# =====================

class PdfsimClient:
    """Pdfsim API 客户端"""

    def __init__(self, base_url: str = "http://127.0.0.1:8000"):
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
                data={
                    "allow_existing": str(allow_existing).lower(),
                    "h3yun_object_id": h3yun_object_id,
                    "h3yun_schema_code": h3yun_schema_code,
                },
                timeout=120,
            )
        r.raise_for_status()
        return r.json()

    def record_import(self, h3yun_object_id: str, h3yun_schema_code: str = "",
                      doc_id: str = "", filename: str = "",
                      status: str = STATUS_PENDING, error_message: str = "",
                      attachment_info: str = "") -> dict:
        r = requests.post(
            f"{self.base_url}/api/v1/import/record",
            params={
                "h3yun_object_id": h3yun_object_id,
                "h3yun_schema_code": h3yun_schema_code,
                "doc_id": doc_id,
                "filename": filename,
                "status": status,
                "error_message": error_message,
            },
            timeout=10,
        )
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


# =====================
# Phase 1: 全量扫描
# =====================

def phase_scan(config: dict, limit: int = 0):
    """
    全量扫描氚云产品，标记每个产品的附件状态
    不碰氚云原数据，状态全写自己PG
    """
    h3 = H3YunClient(
        engine_code=config["h3yun"]["engine_code"],
        engine_secret=config["h3yun"]["engine_secret"],
        base_url=config["h3yun"].get("base_url", "https://www.h3yun.com"),
    )
    pdfsim = PdfsimClient(config["pdfsim_url"])

    schema_code = config["h3yun"]["product_schema_code"]
    attachment_field = config.get("attachment_field_code", "F0000107")
    name_field = config.get("name_field_code", "Name")

    # 检查服务
    try:
        health = pdfsim.health()
        print(f"[info] Pdfsim 服务就绪: {health['docs']} 文档 / {health['vectors']} 向量")
    except Exception as e:
        print(f"[error] Pdfsim 服务不可达: {e}")
        return

    # 拉取全部产品
    print(f"\n[scan] 从氚云加载产品列表 (schema={schema_code})...")
    all_objects = []
    page = 0
    while True:
        try:
            batch = h3.load_biz_objects(schema_code, page_size=200, page_index=page)
        except Exception as e:
            print(f"[error] 加载第{page}页失败: {e}")
            break
        if not batch:
            break
        all_objects.extend(batch)
        print(f"  第{page}页: {len(batch)} 条")
        if len(batch) < 200:
            break
        page += 1
        time.sleep(0.5)

    total = len(all_objects)
    if limit > 0:
        all_objects = all_objects[:limit]

    print(f"\n[scan] 共 {total} 个产品，开始分类标记...\n")

    # 分类统计
    counts = {s: 0 for s in STATUS_LABELS}

    for i, obj in enumerate(all_objects):
        obj_id = obj.get("ObjectId", "")
        name = obj.get(name_field, obj_id)
        attachments = obj.get(attachment_field, "")

        # 判断附件状态
        if not attachments or str(attachments).strip() == "":
            status = STATUS_NO_ATTACHMENT
            file_ids = []
        else:
            file_ids = [f.strip() for f in str(attachments).split(",") if f.strip()]
            if not file_ids:
                status = STATUS_NO_ATTACHMENT
            else:
                # 有附件，标记为 pending（后续入库时再细判断是否非PDF）
                status = STATUS_PENDING

        counts[status] += 1

        # 记录到我们自己的PG
        attachment_info = f"{len(file_ids)}个附件" if file_ids else "无"
        pdfsim.record_import(
            h3yun_object_id=obj_id,
            h3yun_schema_code=schema_code,
            filename=name,
            status=status,
            attachment_info=attachment_info,
        )

        # 进度
        if (i + 1) % 50 == 0 or (i + 1) == len(all_objects):
            print(f"  [{i+1}/{len(all_objects)}] 已扫描...")

        time.sleep(0.1)  # 控制频率

    # 汇总
    print(f"\n{'='*50}")
    print(f"扫描完成！共 {len(all_objects)} 个产品\n")
    print(f"  {'状态':<12} {'数量':>6}  说明")
    print(f"  {'-'*12} {'-'*6}  {'-'*20}")
    for status, count in counts.items():
        if count > 0:
            label = STATUS_LABELS.get(status, status)
            print(f"  {status:<12} {count:>6}  {label}")

    actionable = counts.get(STATUS_PENDING, 0)
    print(f"\n  可入库: {actionable} 个")
    print(f"  需人工处理: {sum(counts[s] for s in TERMINAL_STATUSES)} 个")
    print(f"  （氚云原数据未改动）")


# =====================
# Phase 2: 执行入库
# =====================

def phase_import(config: dict, limit: int = 0, retry: bool = False):
    """
    入库：只处理 pending 和可重试状态的
    """
    h3 = H3YunClient(
        engine_code=config["h3yun"]["engine_code"],
        engine_secret=config["h3yun"]["engine_secret"],
        base_url=config["h3yun"].get("base_url", "https://www.h3yun.com"),
    )
    pdfsim = PdfsimClient(config["pdfsim_url"])

    schema_code = config["h3yun"]["product_schema_code"]
    attachment_field = config.get("attachment_field_code", "F0000107")
    name_field = config.get("name_field_code", "Name")
    download_dir = Path(config.get("download_dir", "/tmp/pdfsim_downloads"))
    download_dir.mkdir(parents=True, exist_ok=True)

    # 检查服务
    try:
        health = pdfsim.health()
        print(f"[info] Pdfsim 服务就绪: {health['docs']} 文档 / {health['vectors']} 向量")
    except Exception as e:
        print(f"[error] Pdfsim 服务不可达: {e}")
        return

    # 重新拉氚云产品列表（需要附件信息）
    print(f"\n[import] 加载氚云产品列表...")
    all_objects = []
    page = 0
    while True:
        try:
            batch = h3.load_biz_objects(schema_code, page_size=200, page_index=page)
        except Exception as e:
            print(f"[error] 加载第{page}页失败: {e}")
            break
        if not batch:
            break
        all_objects.extend(batch)
        if len(batch) < 200:
            break
        page += 1
        time.sleep(0.5)

    # 建立ObjectId -> 附件映射
    obj_map = {}
    for obj in all_objects:
        obj_id = obj.get("ObjectId", "")
        attachments = obj.get(attachment_field, "")
        name = obj.get(name_field, obj_id)
        file_ids = [f.strip() for f in str(attachments).split(",") if f.strip()] if attachments else []
        obj_map[obj_id] = {"name": name, "file_ids": file_ids}

    # 获取待入库/可重试的
    try:
        pending_resp = pdfsim.get_import_pending(limit=10000)
        pending_items = pending_resp.get("items", [])
    except Exception:
        pending_items = []

    if retry:
        try:
            failed_resp = pdfsim.get_import_failed(limit=10000)
            failed_items = failed_resp.get("items", [])
            pending_items.extend(failed_items)
        except Exception:
            pass

    if not pending_items:
        print("[info] 没有待处理的项目")
        return

    print(f"[info] 待入库: {len(pending_items)} 个")

    if limit > 0:
        pending_items = pending_items[:limit]

    # 执行入库
    success = 0
    failed = 0
    skipped = 0

    for i, item in enumerate(pending_items):
        obj_id = item.get("h3yun_object_id", "")
        name = item.get("filename", obj_id)
        obj_info = obj_map.get(obj_id, {})
        file_ids = obj_info.get("file_ids", [])

        if not file_ids:
            # 二次确认无附件
            pdfsim.record_import(
                h3yun_object_id=obj_id,
                h3yun_schema_code=schema_code,
                filename=name,
                status=STATUS_NO_ATTACHMENT,
                error_message="二次确认无附件",
            )
            skipped += 1
            continue

        print(f"\n[{i+1}/{len(pending_items)}] {name} (ObjectId={obj_id})")

        # 标记入库中
        pdfsim.record_import(
            h3yun_object_id=obj_id,
            h3yun_schema_code=schema_code,
            filename=name,
            status=STATUS_INGESTING,
        )

        doc_id = ""
        any_success = False
        last_error = ""

        for file_id in file_ids:
            # 下载
            pdf_path = download_dir / f"{obj_id}_{file_id}.pdf"
            print(f"  下载 {file_id}...", end=" ", flush=True)

            try:
                ok = h3.download_file(file_id, pdf_path)
                if not ok:
                    print("失败")
                    last_error = "下载附件失败"
                    pdfsim.record_import(
                        h3yun_object_id=obj_id,
                        h3yun_schema_code=schema_code,
                        filename=name,
                        status=STATUS_DOWNLOAD_FAIL,
                        error_message=f"附件{file_id}下载失败",
                    )
                    continue
            except Exception as e:
                print(f"异常({e})")
                last_error = f"下载异常: {e}"
                continue

            # 检查是否PDF
            content = pdf_path.read_bytes()
            if len(content) < 10:
                print("空文件")
                last_error = "附件为空文件"
                pdfsim.record_import(
                    h3yun_object_id=obj_id,
                    h3yun_schema_code=schema_code,
                    filename=name,
                    status=STATUS_NON_PDF,
                    error_message=f"附件{file_id}为空文件",
                )
                skipped += 1
                pdf_path.unlink(missing_ok=True)
                continue

            if content[:5] != b"%PDF-":
                ext = pdf_path.suffix
                print(f"非PDF({ext})")
                last_error = f"附件非PDF格式"
                pdfsim.record_import(
                    h3yun_object_id=obj_id,
                    h3yun_schema_code=schema_code,
                    filename=name,
                    status=STATUS_NON_PDF,
                    error_message=f"附件{file_id}非PDF格式",
                )
                skipped += 1
                pdf_path.unlink(missing_ok=True)
                continue

            print("OK", flush=True)

            # 入库
            print(f"  入库...", end=" ", flush=True)
            try:
                result = pdfsim.upload(
                    pdf_path,
                    allow_existing=True,
                    h3yun_object_id=obj_id,
                    h3yun_schema_code=schema_code,
                )
                if result.get("ok"):
                    doc_id = result.get("doc_id", "")
                    pages = result.get("num_pages", 0)
                    msg = result.get("message", "")
                    print(f"OK (doc_id={doc_id}, {pages}页, {msg})")
                    any_success = True
                    success += 1
                else:
                    msg = result.get("message", "unknown")
                    print(f"失败({msg})")
                    last_error = msg
                    failed += 1
            except requests.exceptions.HTTPError as e:
                detail = ""
                try:
                    detail = e.response.json().get("detail", str(e))
                except Exception:
                    detail = str(e)
                print(f"失败({detail})")
                last_error = detail[:200]
                if "解析失败" in detail or "PDF" in detail:
                    pdfsim.record_import(
                        h3yun_object_id=obj_id,
                        h3yun_schema_code=schema_code,
                        filename=name,
                        status=STATUS_PARSE_FAIL,
                        error_message=detail[:500],
                    )
                failed += 1
            except Exception as e:
                print(f"异常({e})")
                last_error = f"入库异常: {e}"
                failed += 1

            pdf_path.unlink(missing_ok=True)

        # 更新最终状态
        if any_success:
            pdfsim.record_import(
                h3yun_object_id=obj_id,
                h3yun_schema_code=schema_code,
                doc_id=doc_id,
                filename=name,
                status=STATUS_DONE,
            )
            print(f"  ✓ 入库完成")
        elif last_error:
            pdfsim.record_import(
                h3yun_object_id=obj_id,
                h3yun_schema_code=schema_code,
                filename=name,
                status=STATUS_FAILED,
                error_message=last_error[:500],
            )
            print(f"  ✗ 入库失败: {last_error}")

        time.sleep(1)

    # 汇总
    print(f"\n{'='*50}")
    print(f"入库完成！")
    print(f"  成功: {success}")
    print(f"  失败: {failed}")
    print(f"  跳过: {skipped}")
    print(f"  （氚云原数据未改动）")


# =====================
# Phase 3: 报告
# =====================

def phase_report(config: dict):
    """查看入库状态报告"""
    pdfsim = PdfsimClient(config["pdfsim_url"])

    try:
        stats = pdfsim.get_import_status()
    except Exception as e:
        print(f"[error] 无法获取入库状态: {e}")
        return

    total = stats.get("total", 0)
    by_status = stats.get("by_status", {})

    print(f"\n{'='*50}")
    print(f"图纸入库状态报告")
    print(f"{'='*50}")
    print(f"  总计: {total} 个产品\n")

    print(f"  {'状态':<16} {'数量':>6}  说明")
    print(f"  {'-'*16} {'-'*6}  {'-'*24}")

    for status in [STATUS_DONE, STATUS_PENDING, STATUS_INGESTING,
                   STATUS_NO_ATTACHMENT, STATUS_NON_PDF, STATUS_PARSE_FAIL,
                   STATUS_DOWNLOAD_FAIL, STATUS_FAILED, STATUS_DUPLICATE]:
        count = by_status.get(status, 0)
        if count > 0:
            label = STATUS_LABELS.get(status, status)
            print(f"  {status:<16} {count:>6}  {label}")

    # 可行动项
    actionable = by_status.get(STATUS_PENDING, 0)
    need_human = sum(by_status.get(s, 0) for s in TERMINAL_STATUSES)
    retryable = sum(by_status.get(s, 0) for s in RETRYABLE_STATUSES)

    print(f"\n  --- 行动项 ---")
    if actionable:
        print(f"  ▸ {actionable} 个待入库，运行: python scripts/h3yun_batch_import.py import")
    if retryable:
        print(f"  ▸ {retryable} 个可重试，运行: python scripts/h3yun_batch_import.py import --retry")
    if need_human:
        print(f"  ▸ {need_human} 个需人工处理:")
        for s in TERMINAL_STATUSES:
            c = by_status.get(s, 0)
            if c:
                print(f"    - {STATUS_LABELS[s]}: {c} 个")


# =====================
# CLI
# =====================

def main():
    parser = argparse.ArgumentParser(description="氚云图纸批量入库（状态写自己PG，不碰氚云）")
    parser.add_argument("action", choices=["scan", "import", "report", "retry"],
                        help="scan=全量扫描, import=执行入库, report=查看报告, retry=重试失败")
    parser.add_argument("--config", default="config.json", help="配置文件路径")
    parser.add_argument("--limit", type=int, default=0, help="限制处理数量（0=全部）")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"[error] 配置文件不存在: {config_path}")
        print("请创建 config.json，示例：")
        print(json.dumps({
            "pdfsim_url": "http://127.0.0.1:8000",
            "h3yun": {
                "engine_code": "your_engine_code",
                "engine_secret": "your_engine_secret",
                "product_schema_code": "D000886XXX",
            },
            "attachment_field_code": "F0000107",
            "name_field_code": "Name",
            "download_dir": "/tmp/pdfsim_downloads",
        }, indent=2, ensure_ascii=False))
        sys.exit(1)

    with open(config_path) as f:
        config = json.load(f)

    if args.action == "scan":
        phase_scan(config, limit=args.limit)
    elif args.action == "import":
        phase_import(config, limit=args.limit, retry=False)
    elif args.action == "retry":
        phase_import(config, limit=args.limit, retry=True)
    elif args.action == "report":
        phase_report(config)


if __name__ == "__main__":
    main()
