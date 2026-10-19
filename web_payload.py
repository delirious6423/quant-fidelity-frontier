"""Project the browser chart payload without changing the public evidence API.

data.json retains full protocols and classifier diagnostics. The browser already
renders the per-metric classification and its evidence, so carrying each point's
complete protocol and each set's duplicate aggregate diagnostics only adds JSON
parsing and memory cost. No claims, coordinates, source links or ranking gates
are removed from the interactive view.
"""


def chart_payload(sets, metrics):
    result = []
    for group in sets:
        item = {key: value for key, value in group.items() if key != 'comparison'}
        item['points'] = [{key: value for key, value in point.items() if key != 'protocol'}
                          for point in group['points']]
        item['comparisons'] = {
            metric: {key: value for key, value in comparison.items() if key != 'failure_examples'}
            for metric, comparison in group.get('comparisons', {}).items()
        }
        result.append(item)
    return {'sets': result, 'metrics': metrics}
