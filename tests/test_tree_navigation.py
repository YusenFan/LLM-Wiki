"""Directory discovery and readable summary filenames, without any search ranking."""
import hashlib
import json
import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

import llm_wiki_bench
from build_summaries import build_summaries, current_summaries, migrate_summary_names, summary_path
from wiki_documents import archive_article, parse_document
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

    def generate(self, title='Shared film history'):
        return Mock(return_value={'title': title, 'description': 'Group overview',
                                  'tags': [], 'summary': 'Related films and people.'})

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

    def test_readable_summary_names_handle_collisions_and_cache_by_membership(self):
        self.groups()
        generated = self.generate()
        self.assertEqual(build_summaries(self.root, generate=generated)['built'], 2)
        self.assertEqual(set(current_summaries(self.root)), {
            'summaries/shared-film-history.md', 'summaries/shared-film-history-2.md'})
        no_calls = Mock(side_effect=AssertionError('must use cached summaries'))
        self.assertEqual(build_summaries(self.root, generate=no_calls)['cached'], 2)
        no_calls.assert_not_called()
        self.write('film/a.md', '# A\n\n## Related Pages\n- [[film/b]] — related\nChanged.\n')
        self.assertEqual(len(current_summaries(self.root)), 1)
        retriever = WikiRetriever(self.root)
        self.assertEqual(sum(e.get('layer') == 'summaries' for e in retriever.tree()['entries']), 1)

    def test_legacy_summary_migration_preserves_content_and_archives_originals(self):
        self.groups()
        build_summaries(self.root, generate=self.generate())
        original = current_summaries(self.root)
        legacy = {}
        for relative, text in original.items():
            members = parse_document(text)[0]['members']
            digest = hashlib.sha256(json.dumps(members).encode()).hexdigest()
            old = f'summaries/{digest}.md'
            (self.root / relative).rename(self.root / old)
            legacy[old] = text
        self.write('summaries/' + 'f' * 64 + '.md', '# Obsolete\n')
        self.assertEqual(len(current_summaries(self.root)), 2)
        report = migrate_summary_names(self.root)
        self.assertEqual(len(report['renamed']), 2)
        self.assertEqual(len(report['archived_originals']), 3)
        for old, new in report['renamed'].items():
            self.assertEqual((self.root / new).read_text(), legacy[old])
            self.assertFalse((self.root / old).exists())
        for backup in report['archived_originals']:
            self.assertTrue((self.root / backup).is_file())
        self.assertEqual(len(current_summaries(self.root)), 2)
        self.assertFalse(any(re.fullmatch(r'[0-9a-f]{64}', path.stem)
                             for path in (self.root / 'summaries').glob('*.md')))
        self.assertEqual(migrate_summary_names(self.root)['renamed'], {})
        no_calls = Mock(side_effect=AssertionError('migration must preserve cache'))
        self.assertEqual(build_summaries(self.root, generate=no_calls)['cached'], 2)
        index = (self.root / 'summaries/_index.md').read_text()
        self.assertIn('shared-film-history', index)

    def test_summary_titles_produce_safe_short_paths_and_reserved_names_work(self):
        for title in ['../A/B', '主题' * 100, 'index', 'overview', 'log', '???']:
            with self.subTest(title=title):
                path = Path(summary_path(('film/a.md', 'film/b.md'), title))
                self.assertEqual(path.parent.as_posix(), 'summaries')
                self.assertLess(len(path.name.encode()), 255)
                self.assertNotIn(path.name, ['index.md', 'overview.md', 'log.md'])


if __name__ == '__main__':
    unittest.main()
