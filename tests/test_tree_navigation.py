"""Directory discovery and existing summary validation, without any search ranking."""
import json
import tempfile
import unittest
from pathlib import Path

import llm_wiki_bench
from summary_catalog import current_summaries
from summary_fixtures import write_summary
from wiki_documents import archive_article
from wiki_retriever import WikiRetriever, WIKI_TOOL_SCHEMAS


class TreeNavigationTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / 'wiki'
        self.root.mkdir()

    def write(self, name, content):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding='utf-8')

    def groups(self):
        for name, other in [('a', 'b'), ('b', 'a'), ('c', 'd'), ('d', 'c')]:
            self.write(f'film/{name}.md', f'# {name.upper()}\n\n## Related Pages\n- [[film/{other}]] — related\n')

    def test_tree_exposes_titles_and_nested_directories_without_index_files(self):
        self.write('film/ed_wood.md', '# Ed Wood\n\n> American filmmaker.\n')
        self.write('film/ed_wood_film.md', '# Ed Wood (film)\n\n> American film.\n')
        self.write('people/directors/scott.md', '# Scott Derrickson\n')
        self.write('.build/private.md', '# Private\n')
        self.write('sources/digests/old.md', '# Old digest\n')
        retriever = WikiRetriever(self.root)
        listing = retriever.tree(depth=1)
        self.assertEqual([e['path'] for e in listing['entries']], ['film', 'people'])
        films = retriever.tree('film')['entries']
        self.assertEqual([e['title'] for e in films], ['Ed Wood', 'Ed Wood (film)'])
        self.assertNotIn('score', json.dumps(films))
        self.assertIn('people/directors', retriever.wiki_map())
        self.assertEqual(retriever.tree('people/directors')['entries'][0]['title'], 'Scott Derrickson')
        self.assertEqual({tool['function']['name'] for tool in WIKI_TOOL_SCHEMAS},
                         {'summary_search', 'wiki_tree', 'wiki_read', 'source_read'})
        self.assertFalse(hasattr(retriever, 'search'))

    def test_tree_pagination_covers_every_file_and_validates_boundaries(self):
        for i in range(7):
            self.write(f'entities/entity-{i}.md', f'# Entity {i}\n')
        retriever = WikiRetriever(self.root)
        first = retriever.tree('entities', depth=1, limit=3)
        second = retriever.tree('entities', depth=1, offset=first['next_offset'], limit=3)
        last = retriever.tree('entities', depth=1, offset=second['next_offset'], limit=3)
        paths = [e['path'] for page in (first, second, last) for e in page['entries']]
        self.assertEqual(len(set(paths)), 7)
        self.assertEqual(first['total'], 7)
        self.assertIsNone(last['next_offset'])
        for args in ({'path': '../secret'}, {'path': '/tmp'}, {'path': 'entities/entity-1.md'},
                     {'depth': 0}, {'offset': -1}, {'limit': 201}):
            with self.subTest(args=args), self.assertRaises(ValueError):
                retriever.tree(**args)

    def test_original_archive_titles_are_visible_without_changing_versions(self):
        source = Path(self.tmp.name) / 'source.md'
        source.write_text('---\ntitle: Ed Wood\n---\n\n# Ed Wood\n\nAn American filmmaker.\n')
        article = archive_article(self.root, source)
        retriever = WikiRetriever(self.root)
        entry = retriever.tree('sources/articles')['entries'][0]
        self.assertEqual(entry['title'], 'Ed Wood')
        self.assertEqual(entry['path'], article['article'])
        self.assertEqual(retriever.source_read(entry['path'])['version'], article['version'])

    def test_existing_summaries_exclude_stale_pages_without_writes(self):
        self.groups()
        for label, members in [('first', ('film/a.md', 'film/b.md')),
                               ('second', ('film/c.md', 'film/d.md'))]:
            write_summary(self.root, f'summaries/{label}.md', members, title='Shared film history')
        self.assertEqual(set(current_summaries(self.root)),
                         {'summaries/first.md', 'summaries/second.md'})
        self.write('film/a.md', '# A\n\n## Related Pages\n- [[film/b]] — related\nChanged.\n')
        before = {p: p.read_bytes() for p in self.root.rglob('*.md')}
        self.assertEqual(set(current_summaries(self.root)), {'summaries/second.md'})
        retriever = WikiRetriever(self.root)
        self.assertEqual(sum(e.get('layer') == 'summaries' for e in retriever.tree()['entries']), 1)
        self.assertEqual(before, {p: p.read_bytes() for p in self.root.rglob('*.md')})

    def test_existing_legacy_summaries_prefer_readable_duplicate(self):
        self.groups()
        legacy = 'summaries/' + 'a' * 64 + '.md'
        content = write_summary(self.root, legacy, ('film/a.md', 'film/b.md'), title='Film history')
        self.assertEqual(set(current_summaries(self.root)), {legacy})
        self.write('summaries/film-history.md', content)
        self.write('summaries/obsolete.md', '# Obsolete\n')
        self.assertEqual(set(current_summaries(self.root)), {'summaries/film-history.md'})


if __name__ == '__main__':
    unittest.main()
