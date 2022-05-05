#!/usr/bin/env python3
from code_to_test import *


def test_get_split_data():
    assert get_split_data('split') == None, "Incorrect function parameters"
    assert type(get_split_data('test')
                ) == DialogueDataset, "Incorrect data type"


def test_dialogue_as_list():
    test = get_split_data('test')

    dialogue = None
    for d in test:
        dialogue = d
        break

    assert type(dialogue) == Dialogue, "Incorrect data type"

    phrases_counter = 0
    for phr in dialogue:
        phrases_counter += 1

    phrases, inds = dialogue_as_list(dialogue)
    assert type(phrases) == list, "Incorrect data type"
    assert type(inds) == list, "Incorrect data type"
    assert len(phrases) == phrases_counter, "Wrong len"
    assert len(phrases) == len(
        inds), "phrases and inds must be of equal length"
    for phrase in phrases:
        assert type(phrase) == str, "Phrases must be of str type"


def test_label_dialoguedataset():
    test = get_split_data('test')

    dialogues = []
    counter = 2
    for d in test:
        dialogues.append(d)
        counter -= 1
        if counter == 0:
            break

    labeled_dialogues = label_dialoguedataset(dialogues)
    assert type(labeled_dialogues) == dict, "Incorrect data type"
    assert type(labeled_dialogues['dialogues']) == list, "Incorrect data type"

    for d in labeled_dialogues['dialogues']:
        assert type(d) == dict, "Incorrect data type"

        for turn in d['turns']:
            assert type(turn['speaker']) == str, "Incorrect data type"
            assert type(turn['utterance']) == str, "Incorrect data type"
            assert type(turn['sentiment']) == dict, "Incorrect data type"
            assert type(turn['emotion']) == dict, "Incorrect data type"

    # Creating my own dialogue to check correct filling
    uttrs = [Utterance(utterance="hi", speaker="None", turn_id=1),
             Utterance(utterance="hello", speaker="None", turn_id=2),
             Utterance(utterance="how are you?", speaker="None", turn_id=3),
             Utterance(utterance="fine", speaker="None", turn_id=4)]
    id = 123
    my_dialogue = Dialogue(utterances=uttrs, dialogue_id=id)
    my_labeled_dialogue = label_dialoguedataset([my_dialogue])

    assert len(my_labeled_dialogue['dialogues']) == 1, "No dialogue info"
    assert len(my_labeled_dialogue['dialogues']
               [0]['turns']) == 4, "Utterances missed"

    assert my_labeled_dialogue['dialogues'][0]['turns'][2]['utterance'] == "how are you?", "Wrong phrase"
    assert my_labeled_dialogue['dialogues'][0]['turns'][2]['turn_id'] == 3, "Wrong turn_id"
    assert my_labeled_dialogue['dialogues'][0]['turns'][2]['speaker'] == "None", "Wrong speaker"
    assert my_labeled_dialogue['dialogues'][0]['turns'][2]['sentiment']['NEU'] > 0, "No NEU label"
