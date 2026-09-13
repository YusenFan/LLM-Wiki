import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from llm_wiki_bench.wiki_store import SourceStore, FactStore, digest, fact_state, frontmatter
from llm_wiki_bench.wiki_retriever import WikiRetriever
from llm_wiki_bench.build_agent import BuildAgent
from llm_wiki_bench.wiki_agent import WikiAgent
from llm_wiki_bench.preprocess_bench import process_hotpotqa, process_musique, process_2wikimhqa
from llm_wiki_bench.validate_wiki import audit
from llm_wiki_bench import bench_config as config


def call(name, args, ident='call'):
    return {'id': ident, 'type': 'function', 'function': {'name': name, 'arguments': json.dumps(args)}}


def message(*calls):
    return {'role': 'assistant', 'content': None, 'tool_calls': list(calls)}


class ToolsTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.sources, self.facts = SourceStore(self.root), FactStore(self.root)
        self.ref = self.sources.archive('urn:source:alice', 'Alice was born in 1980. Alice did not move before 2000.', 'Alice')
        self.cite = {k: self.ref[k] for k in ('source_id', 'version_id')}
        self.cite.update(start=0, end=24, quote='Alice was born in 1980. ')

    def tearDown(self):
        self.temp.cleanup()

    def fact(self, statement='Alice was born in 1980.', **kwargs):
        return {'statement': statement, 'event_time': '1980', 'conditions': '', 'polarity': 'positive',
                'certainty': 'asserted', 'kind': 'fact', 'citations': [self.cite], **kwargs}

    def apply(self, path='people/alice.md', revision=None, **kwargs):
        return self.facts.apply(path, revision, 'person:alice:born1980', [self.fact()], title='Alice', **kwargs)

    def test_versions_and_long_original_survive(self):
        text = '前文\n' * 6000 + 'tail evidence'
        first = self.sources.archive('urn:long', text)
        second = self.sources.archive('urn:long', text + '!')
        self.assertEqual(first['source_id'], second['source_id'])
        self.assertNotEqual(first['version_id'], second['version_id'])
        self.assertEqual(self.sources.get(first['source_id'], first['version_id'])[1], text)
        tail = self.sources.read(first['source_id'], first['version_id'], len(text)-13)
        self.assertEqual(tail['text'], 'tail evidence')

    def test_source_tampering_rejected(self):
        (self.root / self.ref['path']).write_text('changed')
        with self.assertRaisesRegex(ValueError, 'integrity'):
            self.sources.validate_citation(self.cite)

    def test_local_update_preserves_legacy_and_facts(self):
        page = self.root / 'people/alice.md'; page.parent.mkdir()
        legacy = '# Alice\n\n' + 'Unrelated old fact.\n' * 2000
        page.write_text(legacy)
        applied = self.apply(revision=digest(legacy))
        self.facts.apply('people/alice.md', applied['revision'], 'person:alice:born1980',
                         [self.fact('Alice did not move before 2000.', event_time='before 2000', polarity='negative', conditions='before 2000')])
        text = page.read_text()
        self.assertTrue(text.startswith(legacy))
        self.assertEqual(len(fact_state(text)['facts']), 2)
        self.assertEqual(fact_state(text)['facts'][1]['polarity'], 'negative')

    def test_stale_revision_and_identity_mismatch(self):
        result = self.apply()
        with self.assertRaisesRegex(ValueError, 'stale'):
            self.apply()
        with self.assertRaisesRegex(ValueError, 'identity'):
            self.facts.apply('people/alice.md', result['revision'], 'different-person', [self.fact()])

    def test_invalid_quote_and_link_leave_page_untouched(self):
        result = self.apply()
        old = (self.root/'people/alice.md').read_text()
        with self.assertRaises(ValueError):
            self.facts.apply('people/alice.md', result['revision'], 'person:alice:born1980', [self.fact(citations=[{**self.cite, 'quote': 'invented'}])])
        with self.assertRaises(ValueError):
            self.apply(revision=result['revision'], links=['people/missing.md'])
        self.assertEqual((self.root/'people/alice.md').read_text(), old)

    def test_paths_cannot_escape(self):
        for path in ('../escape.md', '/tmp/escape.md', '.hidden/a.md'):
            with self.assertRaises(ValueError):
                self.apply(path)
        (self.root/'outside').symlink_to('/tmp', target_is_directory=True)
        with self.assertRaises(ValueError):
            self.apply('outside/escape.md')
        result = json.loads(WikiRetriever(self.root).execute_tool('wiki_read', {'paths': ['../escape.md']}))
        self.assertIn('error', result)

    def test_add_evidence_and_conflict_retains_old(self):
        first = self.apply()
        second = self.facts.apply('people/alice.md', first['revision'], 'person:alice:born1980',
                                 [self.fact('Alice was born at another time.', conflicts_with=first['fact_ids'])])
        self.apply(revision=second['revision'])
        facts = fact_state((self.root/'people/alice.md').read_text())['facts']
        self.assertEqual(len(facts), 2)
        self.assertEqual(len(facts[0]['citations']), 1)
        self.assertEqual(facts[1]['conflicts_with'], first['fact_ids'])

    def test_live_tree_read_sections_yaml_and_fences(self):
        retriever = WikiRetriever(self.root); retriever.load()
        p = self.root/'untracked/new/page.md'; p.parent.mkdir(parents=True)
        p.write_text('```markdown\n---\ntitle: New Name\naliases:\n  - Exact Alias\ntags:\n  - science\n---\n# Title\n## A\nfirst\n### Detail\nnext\n## B\nlast\n```\n')
        self.assertIn('untracked/new', retriever.wiki_map())
        self.assertEqual(retriever.search('Exact Alias', mode='exact')[0].page.name, 'New Name')
        read = retriever.read(['untracked/new/page.md'], section='A')[0]
        self.assertIn('next', read['text']); self.assertNotIn('last', read['text'])
        listing = retriever.read(['untracked/new'])[0]
        self.assertEqual(listing['pages'][0]['path'], 'untracked/new/page.md')
        p.unlink()
        self.assertEqual(retriever.search('Exact Alias', mode='exact'), [])

    def test_bm25_no_substring_bonuses_and_homonyms(self):
        self.apply()
        p = self.root/'people/other.md'; p.write_text('---\ntitle: Alice\n---\n# Other person')
        p = self.root/'people/annex.md'; p.write_text('# Annex\nannexation')
        retriever = WikiRetriever(self.root)
        self.assertEqual(retriever.search('ann'), [])
        self.assertEqual(len([r for r in retriever.search('Alice', mode='exact') if r.page.dir_name == 'people']), 2)

    def test_paged_reads_do_not_lose_tail(self):
        p = self.root/'people/long.md'; p.parent.mkdir(); text = 'abc' * 10000; p.write_text(text)
        retriever = WikiRetriever(self.root); parts=[]; start=0
        while start is not None:
            read = retriever.read(['people/long.md'], start=start)[0]
            parts.append(read['text']); start=read['next_start']
        self.assertEqual(''.join(parts), text)

    def test_repair_backs_up_and_does_not_guess_ambiguous_links(self):
        p=self.root/'people/a.md';p.parent.mkdir();p.write_text('```markdown\n---\naliases: [Example]\n---\n[[unique]] [[duplicate]]\n```\n')
        (p.parent/'unique.md').write_text('# Unique')
        (p.parent/'duplicate.md').write_text('# A')
        (self.root/'elsewhere').mkdir();(self.root/'elsewhere/duplicate.md').write_text('# B')
        original=p.read_text();report=audit(self.root, repair=True)
        self.assertEqual(report['repaired_pages'], 1)
        self.assertEqual(Path(report['backups'][0]).read_text(), original)
        self.assertIn('[[people/unique]]', p.read_text())
        self.assertIn('[[duplicate]]', p.read_text())
        self.assertTrue(any(i.get('target') == 'duplicate' and not i['repaired'] for i in report['issues']))

    def test_preprocess_preserves_title_variants_and_collisions(self):
        fixture = [dict(_id='1', question='q', answer='a', context=[['A/B',['old']], ['AB',['other']], ['A/B',['new']]])]
        file=self.root/'data.json';file.write_text(json.dumps(fixture))
        for process in (process_hotpotqa, process_2wikimhqa):
            out=self.root/process.__name__
            stats=process(file,out,self.root/'qa')
            self.assertEqual(stats['paragraph_count'],3)
            docs=[p.read_text() for p in (out/'articles').glob('*.md')]
            self.assertTrue(any('old' in t for t in docs));self.assertTrue(any('new' in t for t in docs))
        musique=self.root/'musique.jsonl'
        musique.write_text(json.dumps(dict(id='1',question='q',answer='a',paragraphs=[dict(title='A',paragraph_text=t) for t in ['x','y']]))+'\n')
        self.assertEqual(process_musique(musique,self.root/'musique',self.root/'qa')['paragraph_count'],2)

    def test_answer_requests_more_evidence_and_validates_final(self):
        page = self.apply()
        calls = [message(call('wiki_read', {'paths':['people/alice.md']})),
                 message(call('evidence_note', {'gaps':['Need original birth evidence'], 'new_evidence':[]})),
                 message(call('source_read', {k:self.ref[k] for k in ('source_id','version_id')})),
                 message(call('finish_answer', {'answer':'1980','citations':[self.cite],'reasoning':'Original birth statement.', 'evidence_gaps':[], 'status':'found'}))]
        seen=[]
        def fake(messages, **kwargs):
            seen.append(kwargs['model']); return calls.pop(0)
        result=WikiAgent(WikiRetriever(self.root), call_llm_with_tools=fake, t_max=3).retrieve('When was Alice born?')
        self.assertEqual(result.answer,'1980');self.assertEqual(result.total_calls,3)
        self.assertTrue(all(m==config.LLM_PREMIUM_MODEL for m in seen))
        self.assertEqual(result.citations,[self.cite])

    def test_unread_citation_rejected_and_budget_honored(self):
        finish={'answer':'1980','citations':[self.cite],'reasoning':'Claim','evidence_gaps':[], 'status':'found'}
        calls=[message(call('wiki_search', {'query':'Alice'}),call('wiki_read',{'paths':['people/missing.md']},'excess')),
               message(call('finish_answer',finish)),
               message(call('finish_answer',{'answer':'unknown','citations':[], 'reasoning':'No original read', 'evidence_gaps':['Original not verified'], 'status':'budget_exhausted'}))]
        result=WikiAgent(WikiRetriever(self.root),call_llm_with_tools=lambda *a,**k:calls.pop(0),t_max=1).retrieve('year?')
        self.assertEqual(result.answer,'unknown');self.assertEqual(result.total_calls,1)
        self.assertTrue(any('error' in t['result'] for t in result.tool_calls))

    def test_duplicate_calls_are_not_executed_again(self):
        calls=[message(call('wiki_search',{'query':'xyz'})),message(call('wiki_search',{'query':'xyz'})),
               message(call('finish_answer',{'answer':'unknown','citations':[],'reasoning':'Nothing found', 'evidence_gaps':['Search scope empty'], 'status':'budget_exhausted'}))]
        result=WikiAgent(WikiRetriever(self.root),call_llm_with_tools=lambda *a,**k:calls.pop(0),t_max=2).retrieve('xyz')
        self.assertIn('duplicate',result.tool_calls[1]['result']['error'])

    def test_build_receipt_and_deleted_product_retry(self):
        article=self.root/'input.md';article.write_text('Alice was born in 1980. ')
        ref=self.sources.archive(article.resolve().as_uri(),article.read_text(),'input')
        citation={k:ref[k] for k in ('source_id','version_id')};citation.update(start=0,end=24,quote=article.read_text())
        invocations=[]
        def fake(messages,**kwargs):
            invocations.append(True)
            history=[m for m in messages if m['role']=='tool']
            if not history:
                return message(call('source_read',{k:ref[k] for k in ('source_id','version_id')}))
            if len(history)==1:
                return message(call('fact_apply',{'path':'people/build.md','expected_revision':None,'identity':'alice','facts':[self.fact(citations=[citation])]}))
            return message(call('finish_document',{'summary':'Birth stored','unresolved':[]}))
        builder=BuildAgent(self.root,fake,budget=5)
        result=builder.ingest(article)
        self.assertEqual(result['status'],'complete',result)
        self.assertTrue(builder.ingest(article)['cached'])
        self.assertEqual(len(invocations),3)
        (self.root/'people/build.md').unlink()
        self.assertEqual(builder.ingest(article)['status'],'complete')
        self.assertEqual(len(invocations),6)

    def test_partial_build_cannot_cache_success(self):
        article=self.root/'input.md';article.write_text('x'*20000)
        def fake(messages,**kwargs):
            return message(call('finish_document',{'summary':'pretend complete','unresolved':[]}))
        builder=BuildAgent(self.root,fake,budget=2)
        result=builder.ingest(article)
        self.assertEqual(result['status'],'partial')
        self.assertFalse(builder.ingest(article).get('cached',False))
        self.assertEqual(self.sources.get(result['source_id'],result['version_id'])[1],article.read_text())

    def test_isolated_subtask_shares_budget_and_returns_evidence(self):
        self.apply()
        parent_calls = 0
        def fake(messages, **kwargs):
            nonlocal parent_calls
            task = json.loads(messages[1]['content'])
            is_child = task.get('scope') is not None
            history = [m for m in messages if m['role'] == 'tool']
            if is_child:
                # The child does not inherit the parent's messages or question.
                self.assertNotIn('parent secret context', messages[1]['content'])
                self.assertTrue(all(t['function']['name'] != 'verify_subtask' for t in kwargs['tools']))
                if not history:
                    return message(call('wiki_read', {'paths':['people/alice.md']}))
                if len(history) == 1:
                    return message(call('source_read', {k:self.ref[k] for k in ('source_id','version_id')}))
            elif not history:
                return message(call('verify_subtask', {'why':'check year','purpose':'verify evidence', 'goal':'find birth year',
                                                       'claim':'Alice was born in 1980', 'scope':['people'], 'budget':2}))
            return message(call('finish_answer', {'answer':'1980', 'citations':[self.cite], 'reasoning':'Original supports year', 'evidence_gaps':[], 'status':'found'}))
        result = WikiAgent(WikiRetriever(self.root),call_llm_with_tools=fake,t_max=3,allow_subtasks=True).retrieve('parent secret context')
        self.assertEqual(result.answer,'1980',result.tool_calls)
        self.assertEqual(result.total_calls,3)
        self.assertEqual(result.llm_calls,5)

    def test_child_scope_prevents_outside_reads(self):
        self.apply()
        calls = [message(call('wiki_read', {'paths':['sources/articles/secret.md']})),
                 message(call('finish_answer', {'answer':'unknown','citations':[], 'reasoning':'outside scope', 'evidence_gaps':['Need another scope'], 'status':'budget_exhausted'}))]
        result=WikiAgent(WikiRetriever(self.root),call_llm_with_tools=lambda *a,**k:calls.pop(0),t_max=1,scope=['people'],allow_subtasks=False).retrieve('claim')
        self.assertIn('outside subtask scope',result.tool_calls[0]['result']['error'])

    def test_usage_is_recorded_without_polluting_messages(self):
        calls=0
        def fake(messages,**kwargs):
            nonlocal calls
            self.assertTrue(all('_usage' not in m for m in messages))
            calls+=1
            msg=message(call('wiki_tree',{})) if calls==1 else message(call('finish_answer',{'answer':'unknown','citations':[], 'reasoning':'empty','evidence_gaps':['Missing evidence'],'status':'budget_exhausted'}))
            msg['_usage']={'prompt_tokens':10,'completion_tokens':5,'total_tokens':15}
            return msg
        result=WikiAgent(WikiRetriever(self.root),call_llm_with_tools=fake,t_max=1).retrieve('q')
        self.assertEqual(result.usage_by_model[config.LLM_PREMIUM_MODEL]['total_tokens'],30)

    def test_legacy_source_snapshot_and_metadata_dates_are_readable(self):
        p=self.root/'sources/articles/legacy.md';p.parent.mkdir(parents=True)
        p.write_bytes(b'---\nsource_date: 2012-13-01\n---\nOriginal.\r\n')
        retriever=WikiRetriever(self.root)
        page=json.loads(retriever.execute_tool('wiki_read',{'paths':['sources/articles/legacy.md']}))[0]
        self.assertEqual(page['meta']['source_date'],'2012-13-01')
        ref=page['source_refs'][0]
        self.assertTrue(self.sources.get(ref['source_id'],ref['version_id'])[1].endswith('\r\n'))

    def test_relation_requires_real_targets(self):
        self.apply()
        with self.assertRaisesRegex(ValueError,'two distinct'):
            self.facts.apply('relations/test.md',None,'relation:test',[self.fact(kind='relation')],links=['people/alice.md'])
        (self.root/'people/bob.md').write_text('# Bob')
        result=self.facts.apply('relations/test.md',None,'relation:test',[self.fact(kind='relation')],links=['people/alice.md','people/bob.md'])
        self.assertEqual(len(result['links']),2)

    def test_no_sample_taxonomy_or_purpose_llm_on_init(self):
        with patch.object(config,'WIKI_DIR',self.root/'empty'), patch.object(config,'_current_dataset','test'), patch.object(config,'auto_init_purpose',side_effect=AssertionError('sampling forbidden')):
            config.ensure_wiki_dirs()
            self.assertEqual(config.get_page_types(),{})
            self.assertEqual(config.get_purpose_file(),config.CONFIGS_DIR/'purpose_bench.md')


if __name__=='__main__':
    unittest.main()
