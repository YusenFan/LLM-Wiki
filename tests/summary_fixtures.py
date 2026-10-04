"""Static existing-summary fixtures for retrieval tests."""
from summary_catalog import group_fingerprint
from wiki_documents import knowledge_pages, render_document


def write_summary(root, path, members, *, title='Overview', body='Existing navigation.', tags=()):
    members = tuple(sorted(members))
    text = render_document(
        {'type': 'summary', 'schema': 'related-summary-v1', 'title': title,
         'tags': list(tags), 'members': list(members),
         'fingerprint': group_fingerprint(members, knowledge_pages(root))},
        f'# {title}\n\n{body}\n\n## Member Pages\n'
        + '\n'.join(f'- [[{member[:-3]}]]' for member in members) + '\n')
    target = root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding='utf-8')
    return text
