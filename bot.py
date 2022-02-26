from df_engine.core.keywords import GLOBAL, TRANSITIONS, RESPONSE
from df_engine.core import Context, Actor
import df_engine.conditions as cnd
import df_engine.labels as lbl
from typing import Union, Any
import re


def greeting_lower_case_condition(ctx: Context, actor: Actor, *args, **kwargs) -> bool:
    request = ctx.last_request
    return ("hi" in request.lower()) or ("hello" in request.lower())


def bye_lower_case_condition(ctx: Context, actor: Actor, *args, **kwargs) -> bool:
    request = ctx.last_request
    return "bye" in request.lower()


def nice_to_meet_you_name(ctx: Context, actor: Actor, *args, **kwargs) -> Any:
    request = ctx.last_request
    return "Nice to meet you, " + request + "! Wanna chat?"


plot = {
    "global_flow": {
        "start_node": {
            RESPONSE: "",
            TRANSITIONS: {
                ("greeting_flow", "node1"): cnd.any([greeting_lower_case_condition]),
                lbl.to_fallback(): cnd.true(),
            },
        },
        "fallback_node": {
            RESPONSE: "I do not understand you. Try again.",
            TRANSITIONS: {
                ("greeting_flow", "node1"): cnd.any([greeting_lower_case_condition]),
                ("greeting_flow", "end_node"): cnd.any([bye_lower_case_condition]),
                lbl.repeat(): cnd.true(),
            },
        },
    },
    "greeting_flow": {
        "node1": {RESPONSE: "Hi! What's your name?", TRANSITIONS: {lbl.forward(): cnd.true()}},
        "node2": {
            RESPONSE: nice_to_meet_you_name,
            TRANSITIONS: {
                ("chat_flow", "node1"): cnd.regexp(r"yes|yep", re.IGNORECASE),
                lbl.forward(): cnd.regexp(r"no|later|not now", re.IGNORECASE),
                "end_node": cnd.any([bye_lower_case_condition]),
                lbl.to_fallback(): cnd.true(),
            },
        },
        "node3": {
            RESPONSE: "That's sad :( Bye, then?",
            TRANSITIONS: {
                lbl.forward(): cnd.any([bye_lower_case_condition, cnd.regexp(r"yes|yep", re.IGNORECASE)]),
                lbl.to_fallback(): cnd.true(),
            },
        },
        "end_node": {
            RESPONSE: "Bye!",
            TRANSITIONS: {"node1": cnd.any([greeting_lower_case_condition]), lbl.to_fallback(): cnd.true()},
        },
    },
    "chat_flow": {
        "node1": {RESPONSE: "Sorry, too tired to chat now", TRANSITIONS: {("greeting_flow", "end_node"): cnd.true()}}
    },
}

actor = Actor(plot, start_label=("global_flow", "start_node"), fallback_label=("global_flow", "fallback_node"))


def turn_handler(in_request: str, ctx: Union[Context, dict], actor: Actor):
    ctx = Context.cast(ctx)
    ctx.add_request(in_request)
    ctx = actor(ctx)
    out_response = ctx.last_response
    return out_response, ctx


ctx = {}
while True:
    in_request = input("Your turn: ")
    out_response, ctx = turn_handler(in_request, ctx, actor)
    print("My turn:", out_response, sep=" ")
