"""Собирает tornado_solution.ipynb из tornado_solution.py (ячейки разделены маркерами `# %%`)."""
import json
from pathlib import Path

here = Path(__file__).parent
src = (here / 'tornado_solution.py').read_text(encoding='utf-8')

cells = []
for chunk in src.split('# %%')[1:]:
    is_md = chunk.startswith(' [markdown]')
    body = chunk.split('\n', 1)[1].strip('\n') if '\n' in chunk else ''
    if is_md:
        lines = [l[2:] if l.startswith('# ') else l.lstrip('#') for l in body.splitlines()]
        cells.append({'cell_type': 'markdown', 'metadata': {}, 'source': '\n'.join(lines)})
    else:
        body = '\n'.join(l[2:] if l.startswith(('# %pip', '# !')) else l for l in body.splitlines())
        cells.append({'cell_type': 'code', 'metadata': {}, 'execution_count': None,
                      'outputs': [], 'source': body})

nb = {'cells': cells, 'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python',
                                                  'name': 'python3'},
                                   'language_info': {'name': 'python'}},
      'nbformat': 4, 'nbformat_minor': 5}
(here / 'tornado_solution.ipynb').write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding='utf-8')
print(len(cells), 'cells')
