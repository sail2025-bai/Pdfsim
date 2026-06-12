"""
氚云图纸批量入库脚本

流程：
  1. 从氚云产品表单获取所有图纸附件
  2. 检查"入库状态"字段，跳过已入库的
  3. 下载PDF → 调Pdfsim API入库
  4. 入库成功 → 更新氚云状态为"已入库"
  5. 入库失败 → 状态标记为"入库失败"，记录原因

使用方式：
  python scripts/h3yun_batch_import.py [--config config.json] [--dry-run] [--limit N]

配置文件 config.json：
{
  "pdfsim_url": "http://127.0.0.1:8000",
  "h3yun": {
    "engine_code": "xxx",
    "engine_secret": "xxx",
    "app_code": "D000886JXCGL",
    "product_schema_code": "D000886..."
  }
}
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import urljoin

import requests

# =====================
# 氚云 API 封装
# =====================

class H3YunClient:
    """氚云 OpenAPI 简易客户端"""

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
                         page_index: int = 0, filter: str = "") -> List[dict]:
        """加载业务对象列表"""
        payload = {
            "SchemaCode": schema_code,
            "Filter": filter,
            "IsGetSummary": False,
            "PageIndex": page_index,
            "PageSize": page_size,
        }
        result = self._invoke("LoadBizObjects", payload)
        return result if isinstance(result, list) else []

    def load_biz_object(self, schema_code: str, object_id: str) -> dict:
        """加载单个业务对象"""
        payload = {
            "SchemaCode": schema_code,
            "ObjectId": object_id,
        }
        return self._invoke("LoadBizObject", payload)

    def update_biz_object(self, schema_code: str, object_id: str,
                          fields: dict) -> bool:
        """更新业务对象字段"""
        payload = {
            "SchemaCode": schema_code,
            "ObjectId": object_id,
            "DynamicPropertyList": [
                {"Name": k, "Value": v} for k, v in fields.items()
            ],
        }
        # 注意：氚云OpenAPI可能不支持UpdateBizObject
        # 实际可能需要用UpdateSubRowQuote或C#接口
        try:
            result = self._invoke("UpdateBizObject", payload)
            return True
        except Exception as e:
            print(f"  [warn] 更新氚云对象失败: {e}")
            return False

    def download_file(self, file_id: str, save_path: Path) -> bool:
        """下载附件文件"""
        payload = {
            "FileId": file_id,
        }
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
            if resp.status_code == 200:
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
    """Pdfsim 相似性查找 API 客户端"""

    def __init__(self, base_url: str = "http://127.0.0.1:8000"):
        self.base_url = base_url.rstrip("/")

    def health(self) -> dict:
        r = requests.get(f"{self.base_url}/api/v1/health", timeout=10)
        r.raise_for_status()
        return r.json()

    def upload(self, file_path: Path, allow_existing: bool = True) -> dict:
        with open(file_path, "rb") as f:
            r = requests.post(
                f"{self.base_url}/api/v1/documents",
                files={"file": (file_path.name, f, "application/pdf")},
                data={"allow_existing": str(allow_existing).lower()},
                timeout=120,
            )
        r.raise_for_status()
        return r.json()

    def search_by_file(self, file_path: Path, top_k: int = 5) -> dict:
        with open(file_path, "rb") as f:
            r = requests.post(
                f"{self.base_url}/api/v1/search",
                files={"file": (file_path.name, f, "application/pdf")},
                data={"top_k": str(top_k)},
                timeout=60,
            )
        r.raise_for_status()
        return r.json()


# =====================
# 批量入库主逻辑
# =====================

# 入库状态枚举
STATUS_PENDING = "待入库"      # 默认值，新图纸
STATUS_INGESTING = "入库中"    # 正在处理
STATUS_DONE = "已入库"         # 入库成功
STATUS_FAILED = "入库失败"     # 入库失败
STATUS_SKIP = "跳过"           # 非PDF或不符合条件

def run_import(config: dict, dry_run: bool = False, limit: int = 0,
               status_filter: str = ""):
    """
    执行批量导入

    config 结构：
    {
      "pdfsim_url": "http://127.0.0.1:8000",
      "h3yun": {
        "engine_code": "xxx",
        "engine_secret": "xxx",
        "base_url": "https://www.h3yun.com",
        "product_schema_code": "D000886..."
      },
      "status_field_code": "F0000xxx",  // 氚云"入库状态"字段编码
      "attachment_field_code": "F0000xxx",  // 氚云"图纸附件"字段编码
      "download_dir": "/tmp/pdfsim_downloads"
    }
    """
    h3 = H3YunClient(
        engine_code=config["h3yun"]["engine_code"],
        engine_secret=config["h3yun"]["engine_secret"],
        base_url=config["h3yun"].get("base_url", "https://www.h3yun.com"),
    )
    pdfsim = PdfsimClient(config["pdfsim_url"])

    schema_code = config["h3yun"]["product_schema_code"]
    status_field = config.get("status_field_code", "F0000501")  # 默认字段编码
    attachment_field = config.get("attachment_field_code", "F0000107")
    download_dir = Path(config.get("download_dir", "/tmp/pdfsim_downloads"))
    download_dir.mkdir(parents=True, exist_ok=True)

    # 1. 检查 Pdfsim 服务健康
    try:
        health = pdfsim.health()
        print(f"[info] Pdfsim 服务就绪: {health['docs']} 文档 / {health['vectors']} 向量")
    except Exception as e:
        print(f"[error] Pdfsim 服务不可达: {e}")
        return

    # 2. 从氚云加载产品列表
    print(f"[info] 从氚云加载产品列表 (schema={schema_code})...")
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

    print(f"[info] 共加载 {len(all_objects)} 个产品对象")

    # 3. 过滤待入库的
    pending = []
    for obj in all_objects:
        obj_id = obj.get("ObjectId", "")
        status = obj.get(status_field, "")
        attachments = obj.get(attachment_field, "")

        # 状态过滤
        if status_filter and status != status_filter:
            continue
        if status == STATUS_DONE:
            continue
        if not attachments:
            continue

        # 解析附件（可能为逗号分隔的fileId列表）
        file_ids = [f.strip() for f in str(attachments).split(",") if f.strip()]
        if not file_ids:
            continue

        pending.append({
            "object_id": obj_id,
            "status": status or STATUS_PENDING,
            "file_ids": file_ids,
            "name": obj.get("Name", obj_id),
        })

    if limit > 0:
        pending = pending[:limit]

    print(f"[info] 待处理: {len(pending)} 个 (已过滤已入库和无附件)")

    if dry_run:
        for i, p in enumerate(pending):
            print(f"  [{i+1}] {p['name']} | 状态={p['status']} | 附件={len(p['file_ids'])}个")
        print("[dry-run] 仅列出，未执行入库")
        return

    # 4. 逐个入库
    success = 0
    failed = 0
    skipped = 0

    for i, p in enumerate(pending):
        print(f"\n[{i+1}/{len(pending)}] {p['name']} (ObjectId={p['object_id']})")

        for file_id in p["file_ids"]:
            # 下载 PDF
            pdf_path = download_dir / f"{p['object_id']}_{file_id}.pdf"
            print(f"  下载附件 {file_id}...", end=" ")

            try:
                ok = h3.download_file(file_id, pdf_path)
                if not ok:
                    print("失败(下载)")
                    failed += 1
                    continue
            except Exception as e:
                print(f"异常({e})")
                failed += 1
                continue

            # 检查是否为PDF
            if pdf_path.suffix.lower() != ".pdf":
                # 尝试根据内容判断
                header = pdf_path.read_bytes()[:5]
                if header != b"%PDF-":
                    print("跳过(非PDF)")
                    skipped += 1
                    pdf_path.unlink(missing_ok=True)
                    continue

            print("OK")

            # 上传到 Pdfsim
            print(f"  入库 Pdfsim...", end=" ")
            try:
                result = pdfsim.upload(pdf_path, allow_existing=True)
                if result.get("ok"):
                    print(f"OK (doc_id={result['doc_id']}, pages={result['num_pages']}, {result['message']})")
                    success += 1
                else:
                    print(f"失败({result.get('message', 'unknown')})")
                    failed += 1
            except Exception as e:
                print(f"异常({e})")
                failed += 1

            # 清理临时文件
            pdf_path.unlink(missing_ok=True)

        # 更新氚云状态
        new_status = STATUS_DONE if failed == 0 else STATUS_FAILED
        print(f"  更新氚云状态 -> {new_status}")
        h3.update_biz_object(
            schema_code,
            p["object_id"],
            {status_field: new_status},
        )

        # 控制频率
        time.sleep(1)

    # 5. 汇总
    print(f"\n{'='*50}")
    print(f"入库完成！")
    print(f"  成功: {success}")
    print(f"  失败: {failed}")
    print(f"  跳过: {skipped}")
    print(f"  总计: {len(pending)}")


# =====================
# CLI
# =====================

def main():
    parser = argparse.ArgumentParser(description="氚云图纸批量入库 Pdfsim")
    parser.add_argument("--config", default="config.json", help="配置文件路径")
    parser.add_argument("--dry-run", action="store_true", help="仅列出待入库，不执行")
    parser.add_argument("--limit", type=int, default=0, help="限制处理数量（0=全部）")
    parser.add_argument("--status", default="", help="仅处理指定状态的记录")
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
            "status_field_code": "F0000501",
            "attachment_field_code": "F0000107",
            "download_dir": "/tmp/pdfsim_downloads",
        }, indent=2, ensure_ascii=False))
        sys.exit(1)

    with open(config_path) as f:
        config = json.load(f)

    run_import(config, dry_run=args.dry_run, limit=args.limit,
               status_filter=args.status)


if __name__ == "__main__":
    main()
