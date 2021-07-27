from testSettings import *


def post_pars_concept(text):
    conceptName = []
    conceptId = []
    conceptCh = []

    url = 'https://cs.socmedica.com/api/pars/ParsingConcept'
    param = {'key': '9244f7d34ca284b1',
             'lib': [25],
             'text': text
             }

    nat_json = json.dumps(param)
    headers = {'Content-type': 'application/json'}
    response = requests.post(url, nat_json, headers=headers)
    outs = response.json()

    for out in outs['result']:
        conceptName.append(out['nameConcept'])
        conceptId.append(out['idConcept'])
        conceptCh.append(out['chance'])

    return conceptId, conceptName, conceptCh