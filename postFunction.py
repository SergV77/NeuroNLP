from testSettings import *



def post_pars_concept(text, name_class):
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
        if out['nameConcept'] != None:
            conceptName.append(out['nameConcept'])
            conceptId.append(out['idConcept'])
            conceptCh.append(out['chance'])

    big_dict = {}
    list_dict = []
    for out in outs['result']:
        if out['nameConcept'] != None:
            dict_id_name = {}
            dict_id_name[out['idConcept']] = out['nameConcept']
            list_dict.append(dict_id_name)
    big_dict[name_class] = list_dict

    return conceptId, conceptName, conceptCh, list_dict, big_dict, outs['result']