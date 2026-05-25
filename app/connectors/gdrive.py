"""Google Drive connector — index files in a folder via a service account.

Setup (one time, ~5 min):
1. Create a service account in Google Cloud Console.
2. Download its JSON key.
3. Share the Drive folder with the service account's email
   (it looks like 'name@project.iam.gserviceaccount.com').
4. Set GOOGLE_SERVICE_ACCOUNT_JSON in .env to the absolute path of the JSON.

Indexes Google Docs / Sheets / Slides (exported to text) and any text/* file.
Binary files (PDFs etc.) are skipped — use the dedicated connector for those.
"""
from __future__ import annotations

import io
import os
from typing import Iterator

from . import connectors
from .base import Connector, Document


_EXPORT_TYPES = {
    "application/vnd.google-apps.document": "text/plain",
    "application/vnd.google-apps.spreadsheet": "text/csv",
    "application/vnd.google-apps.presentation": "text/plain",
}


@connectors.register("gdrive")
class GoogleDriveConnector(Connector):
    name = "Google Drive (folder or file)"
    description = (
        "Index Drive files via a service account. Needs "
        "GOOGLE_SERVICE_ACCOUNT_JSON to point at a credentials JSON."
    )
    input_kind = "text"
    placeholder = "Folder ID, file ID, or Drive URL"

    @classmethod
    def is_available(cls) -> bool:
        if not os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON", "").strip():
            return False
        try:
            from googleapiclient.discovery import build  # noqa: F401
            from google.oauth2 import service_account  # noqa: F401
            return True
        except ImportError:
            return False

    @staticmethod
    def _extract_id(ref: str) -> str:
        ref = ref.strip()
        # Drive URLs: /folders/<id>, /file/d/<id>, ?id=<id>
        import re
        for pattern in (
            r"/folders/([a-zA-Z0-9_-]+)",
            r"/file/d/([a-zA-Z0-9_-]+)",
            r"[?&]id=([a-zA-Z0-9_-]+)",
        ):
            m = re.search(pattern, ref)
            if m:
                return m.group(1)
        # Otherwise assume the user pasted a raw ID.
        return ref

    def fetch(
        self,
        payload: str,
        source_name: str | None = None,
    ) -> Iterator[Document]:
        from google.oauth2 import service_account
        from googleapiclient.discovery import build
        from googleapiclient.http import MediaIoBaseDownload

        sa_path = os.environ["GOOGLE_SERVICE_ACCOUNT_JSON"]
        creds = service_account.Credentials.from_service_account_file(
            sa_path,
            scopes=["https://www.googleapis.com/auth/drive.readonly"],
        )
        service = build("drive", "v3", credentials=creds, cache_discovery=False)

        target_id = self._extract_id(payload)
        meta = (
            service.files()
            .get(fileId=target_id, fields="id, name, mimeType")
            .execute()
        )

        if meta.get("mimeType") == "application/vnd.google-apps.folder":
            results = (
                service.files()
                .list(
                    q=f"'{target_id}' in parents and trashed=false",
                    fields="files(id, name, mimeType)",
                    pageSize=200,
                )
                .execute()
            )
            files = results.get("files") or []
        else:
            files = [meta]

        for f in files:
            try:
                text = self._read_file(service, f, MediaIoBaseDownload)
            except Exception:
                continue
            if text:
                yield Document(text=text, source=f"gdrive:{f.get('name', f.get('id'))}")

    @staticmethod
    def _read_file(service, f, MediaIoBaseDownload) -> str:
        mime = f.get("mimeType", "")
        file_id = f["id"]
        if mime in _EXPORT_TYPES:
            req = service.files().export_media(
                fileId=file_id, mimeType=_EXPORT_TYPES[mime]
            )
        elif mime.startswith("text/") or mime in (
            "application/json",
            "application/xml",
        ):
            req = service.files().get_media(fileId=file_id)
        else:
            return ""  # skip binary
        buf = io.BytesIO()
        downloader = MediaIoBaseDownload(buf, req)
        done = False
        while not done:
            _, done = downloader.next_chunk()
        return buf.getvalue().decode("utf-8", errors="replace")
