import builtins
import textwrap

from consts import MAX_REQUEST_CHAR_COUNT_FOR_LOG


async def truncate_dict(_dict):
    trunc_request = {}
    for k, v in _dict.items():
        await switch_case_for_type(k, trunc_request, v)

    return trunc_request


async def switch_case_for_type(k, trunc_request, v):
    match type(v):
        case builtins.str:
            trunc_request[k] = await truncate_str(v)
        case builtins.list:
            for item in v:
                await switch_case_for_type(k, trunc_request, item)
        case builtins.dict:
            trunc_request[k] = await truncate_dict(v)


async def truncate_str(v):
    return textwrap.shorten(v, width=MAX_REQUEST_CHAR_COUNT_FOR_LOG, placeholder="...")
