import json


def parse_json(text):
    # No regex extraction, answer recovery, markdown stripping or LLM judge.
    def reject_constant(value):
        raise ValueError(value)
    def unique_keys(pairs):
        obj = {}
        for k,v in pairs:
            if k in obj: raise ValueError('duplicate JSON key')
            obj[k] = v
        return obj
    return json.loads(text, parse_constant=reject_constant, object_pairs_hook=unique_keys)


def grade(item, response):
    if response.get('finish_reason') == 'length':
        return {'status':'truncated','answer':None,'explanation':None}
    if response.get('finish_reason') != 'stop':
        return {'status':'unparsed','answer':None,'explanation':None}
    try:
        data = parse_json(response['content'])
        if not isinstance(data,dict) or set(data) != {'answer','explanation'} or not isinstance(data['answer'],str) or data['answer'] not in item.options or not isinstance(data['explanation'],str):
            raise ValueError()
    except (ValueError, TypeError, KeyError):
        return {'status':'unparsed','answer':None,'explanation':None}
    return {'status':'correct' if data['answer'] == item.answer else 'wrong', **data}


def solver_messages(item, rng):
    options = item.options.copy()
    rng.shuffle(options)
    return [{'role':'system','content':'Answer the statistical question using only its stated information. Return exactly one JSON object with keys answer (one listed option) and explanation (string). No markdown.'}, {'role':'user','content':item.prompt+'\nOptions: '+json.dumps(options)}]


def solve(client, item, messages, record_attempt):
    attempts = []
    for multiplier in (1, 2):
        response = client.complete(messages, max_tokens=client.max_tokens*multiplier)
        attempts.append(response)
        record_attempt(response, len(attempts))
        if response['finish_reason'] != 'length': break
    return {'result':grade(item, attempts[-1]), 'attempts':attempts, 'messages':messages}
