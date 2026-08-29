"""发布说明格式化函数。"""


def build_release_note(version: str, summary: str) -> str:
    """组合版本号与摘要。"""

    return f"{version}: {summary}"
