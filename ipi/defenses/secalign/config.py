"""
SecAlign Configuration: Delimiters, Prompt Formats, and Paper Hyperparameters.
"""

IGNORE_INDEX = -100
DEFAULT_TOKENS = {'pad_token': '[PAD]', 'eos_token': '</s>', 'bos_token': '<s>', 'unk_token': '<unk>'}
TEXTUAL_DELM_TOKENS = ['instruction', 'input', 'response', '###', ':']
SPECIAL_DELM_TOKENS = ['[INST]', '[INPT]', '[RESP]', '[MARK]', '[COLN]']

#: Stripped from the data channel at inference time by ``recursive_filter``.
#: Upstream ``test.py`` applies this for every SecAlign model regardless of the
#: frontend delimiter scheme — with TextTextText the '##' entry is what stops an
#: attacker writing '### instruction:' into the data channel.
FILTERED_TOKENS = SPECIAL_DELM_TOKENS + ['##']

OTHER_DELM_TOKENS = {
    'mark': ['{s}', '|{s}|', '<{s}>', '[{s}]', '<|{s}|>', '[|{s}|]', '<[{s}]>', '\'\'\'{s}\'\'\'', '***{s}***'],
    'inst': ['Command', 'Rule', 'Prompt', 'Task'],
    'inpt': ['Data', 'Context', 'Text'],
    'resp': ['Output', 'Answer', 'Reply'],
    'user': ['', 'Prompter ', 'User ', 'Human '],
    'asst': ['', 'Assistant ', 'Chatbot ', 'Bot ', 'GPT ', 'AI '],
}
OTHER_DELM_FOR_TEST = 2

DELIMITERS = {
    "TextTextText": [
        TEXTUAL_DELM_TOKENS[3] + ' ' + TEXTUAL_DELM_TOKENS[0] + TEXTUAL_DELM_TOKENS[4],
        TEXTUAL_DELM_TOKENS[3] + ' ' + TEXTUAL_DELM_TOKENS[1] + TEXTUAL_DELM_TOKENS[4],
        TEXTUAL_DELM_TOKENS[3] + ' ' + TEXTUAL_DELM_TOKENS[2] + TEXTUAL_DELM_TOKENS[4]
    ],
    "SpclSpclSpcl": [
        SPECIAL_DELM_TOKENS[3] + ' ' + SPECIAL_DELM_TOKENS[0] + SPECIAL_DELM_TOKENS[4],
        SPECIAL_DELM_TOKENS[3] + ' ' + SPECIAL_DELM_TOKENS[1] + SPECIAL_DELM_TOKENS[4],
        SPECIAL_DELM_TOKENS[3] + ' ' + SPECIAL_DELM_TOKENS[2] + SPECIAL_DELM_TOKENS[4]
    ],
    "llama-7b": ['[INST] ', '', ' [/INST]'],
    "Mistral-7B-Instruct-v0.1": ['<s>[INST] ', ' ', ' [/INST]'],
    "Meta-Llama-3-8B-Instruct": [
        '<|begin_of_text|><|start_header_id|>system<|end_header_id|>',
        '<|eot_id|><|start_header_id|>user<|end_header_id|>',
        '<|eot_id|><|start_header_id|>assistant<|end_header_id|>'
    ],
}

SYS_INPUT = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n"
SYS_NO_INPUT = SYS_INPUT.replace(", paired with an input that provides further context", "")

PROMPT_FORMAT = {}
for name, delm in DELIMITERS.items():
    if 'Text' not in name and 'Spcl' not in name:
        sys_input = ''
        sys_no_input = ''
    else:
        sys_input = SYS_INPUT
        sys_no_input = SYS_NO_INPUT
    PROMPT_FORMAT[name] = {}
    PROMPT_FORMAT[name]["prompt_input"] = sys_input + delm[0] + "\n{instruction}\n\n" + delm[1] + "\n{input}\n\n" + delm[2] + "\n"
    PROMPT_FORMAT[name]["prompt_no_input"] = sys_no_input + delm[0] + "\n{instruction}\n\n" + delm[2] + "\n"

PAPER_DATA_URLS = {
    "alpaca_cleaned": "https://raw.githubusercontent.com/gururise/AlpacaDataCleaned/refs/heads/main/alpaca_data_cleaned.json",
    "alpaca_stanford": "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/refs/heads/main/alpaca_data.json",
}

PAPER_LR_CONFIG = {
    'llama-7b': {'lr': '2e-5', 'dpo_lr': '2.0e-4', 'kto_lr': '8e-5', 'orpo_lr': '6.4e-4'},
    'Mistral-7B-v0.1': {'lr': '2.5e-6', 'dpo_lr': '1.4e-4'},
    'Meta-Llama-3-8B': {'lr': '2e-6', 'dpo_lr': '1.6e-4'},
}
