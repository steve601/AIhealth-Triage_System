def to_text(obj):
    """Convert Pydantic objects or other inputs to plain text."""
    if obj is None:
        return ""
    if hasattr(obj, "json"):  # Pydantic model
        return obj.json()
    return str(obj)

def extract_text(docs):
    """Convert list of Document objects into plain text snippets"""
    return "\n\n".join([d.page_content for d in docs]) if docs else ""