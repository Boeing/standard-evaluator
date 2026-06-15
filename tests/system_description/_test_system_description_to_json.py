"""
Test harness for `test_system_description_to_json.py`.
"""
import logging
import numpy as np
import unittest
import os
from openmdao.core.constants import _UNDEFINED
import json

# for pipeline
from standard_evaluator import system_description_to_json as sd
from standard_evaluator import circuit_example as ce


# Module attributes.  ----------------------------------------------------

DEBUG = False
"""Whether to log debugging information and/or execute debugging code."""

RUN_ALL_TESTS = True
"""Whether to run all tests or manually selected tests."""


# Unit test classes.  ----------------------------------------------------

# class TestSystemDescriptionToJson(unittest.TestCase):
class test_system_description_to_json(unittest.TestCase):

    # Infrastructure methods.  -------------------------------------------

    def setUp(self):
        # Define test problem and model
        self.cir = ce.Circuit()
        self.cir.create_model()

        # run n2 to genereate the basic dict
        self.test_dict_basic = sd.model_to_dict(self.cir.p, values=True, title="JDM")

    # Unit test methods.  ------------------------------------------------

    @unittest.skipIf(not RUN_ALL_TESTS, "Temporarily disabled.")
    def test_get_additional_options(self):
        """
        Test `get_additional_options()`.
        """
        ###
        # Configuration.
        test_dict = {"val": 3, "recordable": True}
        add_dict = {"n1": {"additional_option_1": "little", "additional_option_2": "big"},
                    "n2": {"additional_option_1": "little", "additional_option_2": "big"},
                    "D1": {"additional_option_1": "little", "additional_option_2": "big"},
                    "R1": {"additional_option_1": "little", "additional_option_2": "big"}}
        out_dict = {'Optional': {'D1': {'additional_option_1': 'little',
                                        'additional_option_2': 'big'},
                                 'R1': {'additional_option_1': 'little',
                                        'additional_option_2': 'big'},
                                 'n1': {'additional_option_1': 'little',
                                        'additional_option_2': 'big'},
                                 'n2': {'additional_option_1': 'little',
                                        'additional_option_2': 'big'}},
                    'recordable': True,
                    'val': 3}
        add_option_file_ = "additional_options.json"
        with open(add_option_file_, 'w') as adF:
            json.dump(add_dict, adF)

        # Execution.
        test_add_opt = sd.get_additional_options(test_dict, add_option_file_)

        # Validation.
        assert test_add_opt == out_dict

    # TODO: Can't figure out how to get this to work
    @unittest.skipIf(not RUN_ALL_TESTS, "Temporarily disabled.")
    def test__serialize_single_option_big(self):
        """
        Test `serialize_single_option()`.
        """
        ###
        # Configuration.
        test_dict = {"val": "From Wikipedia, the free encyclopedia This article is about Homer's epic poem. For other uses, see Odyssey (disambiguation). 'Homer's Odyssey' redirects here. For The Simpsons episode, see Homer's Odyssey (The Simpsons). Odyssey by Homer 15th-century manuscript of Book I written by scribe John Rhosos (British Museum) Written	c. 8th century BC Language	Homeric Greek Genre(s)	Epic poetry Publisher	1488 Published in English	1614 Lines	12,109 Metre	Dactylic hexameter Full text The Odyssey at Wikisource  The Odyssey (/ˈɒdɪsi/;[1] Ancient Greek: Ὀδύσσεια, romanized: Odýsseia)[2][3] is one of two major ancient Greek epic poems attributed to Homer. It is one of the oldest extant works of literature still widely read by modern audiences. As with the Iliad, the poem is divided into 24 books. It follows the Greek hero Odysseus, king of Ithaca, and his journey home after the Trojan War. After the war, which lasted ten years, his journey from Troy to Ithaca, via Africa and southern Europe, lasted for ten additional years during which time he encountered many perils and all of his crewmates were killed. In his absence, Odysseus was assumed dead, and his wife Penelope and son Telemachus had to contend with a group of unruly suitors who were competing for Penelope's hand in marriage.  The Odyssey was originally composed in Homeric Greek in around the 8th or 7th century BC and, by the mid-6th century BC, had become part of the Greek literary canon. In antiquity, Homer's authorship of the poem was not questioned, but contemporary scholarship predominantly assumes that the Iliad and the Odyssey were composed independently and that the stories formed as part of a long oral tradition. Given widespread illiteracy, the poem was performed by an aoidos or rhapsode and was more likely to be heard than read.  Crucial themes in the poem include the ideas of nostos (νόστος; 'return'), wandering, xenia (ξενία; 'guest-friendship'), testing, and omens. Scholars still reflect on the narrative significance of certain groups in the poem, such as women and slaves, who have a more prominent role in the epic than in many other works of ancient literature. This focus is especially remarkable when contrasted with the Iliad, which centres the exploits of soldiers and kings during the Trojan War.  The Odyssey is regarded as one of the most significant works of the Western canon. The first English translation of the Odyssey was in the 16th century. Adaptations and re-imaginings continue to be produced across a wide variety of media. In 2018, when BBC Culture polled experts around the world to find literature's most enduring narrative, the Odyssey topped the list.[4] Synopsis Exposition (books 1–4) A mosaic depicting Odysseus, from the villa of La Olmeda, Pedrosa de la Vega, Spain, late 4th–5th centuries AD  The Odyssey begins after the end of the ten-year Trojan War (the subject of the Iliad), from which Odysseus (also known by the Latin variant Ulysses), king of Ithaca, has still not returned because he angered Poseidon, the god of the sea. Odysseus' son, Telemachus, is about 20 years old and is sharing his absent father's house on the island of Ithaca with his mother Penelope and the suitors of Penelope, a crowd of 108 boisterous young men who each aim to persuade Penelope for her hand in marriage, all the while reveling in the king's palace and eating up his wealth.  Odysseus' protectress, the goddess Athena, asks Zeus, king of the gods, to finally allow Odysseus to return home when Poseidon is absent from Mount Olympus. Disguised as a chieftain named Mentes, Athena visits Telemachus to urge him to search for news of his father. He offers her hospitality, and they observe the suitors dining rowdily while Phemius, the bard, performs a narrative poem for them.  That night, Athena, disguised as Telemachus, finds a ship and crew for the true prince. The next morning, Telemachus calls an assembly of citizens of Ithaca to discuss what should be done with the insolent suitors, who then scoff at Telemachus. Accompanied by Athena (now disguised as Mentor), the son of Odysseus departs for the Greek mainland to the household of Nestor, most venerable of the Greek warriors at Troy, who resided in Pylos after the war.  From there, Telemachus rides to Sparta, accompanied by Nestor's son. There he finds Menelaus and Helen, who are now reconciled. Both Helen and Menelaus also say that they returned to Sparta after a long voyage by way of Egypt. There, on the island of Pharos, Menelaus encounters the old sea-god Proteus, who tells him that Odysseus was a captive of the nymph Calypso. Telemachus learns the fate of Menelaus' brother, Agamemnon, king of Mycenae and leader of the Greeks at Troy: he was murdered on his return home by his wife Clytemnestra and her lover Aegisthus. The story briefly shifts to the suitors, who have only just realized that Telemachus is gone. Angry, they formulate a plan to ambush his ship and kill him as he sails back home. Penelope overhears their plot and worries for her son's safety. Escape to the Phaeacians (books 5–8) Charles Gleyre, Odysseus and Nausicaä  In the course of Odysseus' seven years as a captive of Calypso on the island Ogygia, she has fallen deeply in love with him, even though he spurns her offers of immortality as her husband and still mourns for home. She is ordered to release him by the messenger god Hermes, who has been sent by Zeus in response to Athena's plea. Odysseus builds a raft and is given clothing, food, and drink by Calypso. When Poseidon learns that Odysseus has escaped, he wrecks the raft, but helped by a veil given by the sea nymph Ino, Odysseus swims ashore on Scherie, the island of the Phaeacians. Naked and exhausted, he hides in a pile of leaves and falls asleep.  The next morning, awakened by girls' laughter, he sees the young Nausicaä, who has gone to the seashore with her maids after Athena told her in a dream to do so. He appeals for help. She encourages him to seek the hospitality of her parents, Arete and Alcinous. Alcinous promises to provide him a ship to return him home without knowing the identity of Odysseus. He remains for several days. Odysseus asks the blind singer Demodocus to tell the story of the Trojan Horse, a stratagem in which Odysseus had played a leading role. Unable to hide his emotion as he relives this episode, Odysseus at last reveals his identity. He then tells the story of his return from Troy.", "recordable": True}
        # This doesn't work
        _MAX_OPTION_SIZE = 1

        # Execution.
        test_big_val = sd._serialize_single_option(test_dict)


    @unittest.skipIf(not RUN_ALL_TESTS, "Temporarily disabled.")
    def test__serialize_single_option_small(self):
        ###
        # Configuration.
        test_dict = {"val": 3, "recordable": True}

        # Execution.
        test_small_val = sd._serialize_single_option(test_dict)

        # Validation.
        assert test_small_val == 3

    @unittest.skipIf(not RUN_ALL_TESTS, "Temporarily disabled.")
    def test__serialize_single_option_recordable(self):
        ###
        # Configuration.
        test_dict = {"val": 3}

        # Execution.
        test_recordable = sd._serialize_single_option(test_dict)

        # Validation.
        assert test_recordable == 'Not Recordable'

    @unittest.skipIf(not RUN_ALL_TESTS, "Temporarily disabled.")
    def test__serialize_single_option_undefined(self):
        ###
        # Configuration.
        test_dict = {"val": _UNDEFINED, "recordable": True}

        # Execution.
        test_undefined = sd._serialize_single_option(test_dict)

        # Validation.
        assert test_undefined == 'UNDEFINED'

    @unittest.skipIf(not RUN_ALL_TESTS, "Temporarily disabled.")
    def test_pull_connections_list(self):
        # (full_model_dict: dict) test_dict_basic
        ###
        # Configuration.
        test_dict = self.test_dict_basic

        # Execution.
        test_connections = sd.pull_connections_list(test_dict)
        print(f'test_connections: {test_connections}')

        # Validation.
        assert test_connections.sort() == ['R2', 'n1', 'n2', 'D1', 'R1'].sort()


    @unittest.skipIf( RUN_ALL_TESTS, "Temporarily disabled.")
    def test_model_to_dict(self):
        ###
        # Configuration.
        # Need better test that actually has partials

        # Execution.
        partial_list = sd._get_declare_partials(self.cir.model)
        json_vars = sd.model_to_dict(self.cir.model,
                          path=None,
                          values=_UNDEFINED,
                          case_id=None,
                          title="JDM")
        sd_name = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(sd_name, "reference_model.json"), "w") as f:
            json_reference = json.dump(f, json_vars)

        # Validation.
        json_reference = []
        # sd_name = "C:/Users/MUSIAKJ/Documents/ApMath/2024_work/NASA_MBSA/std_evaluator/standard_evaluator/src/system_description/"
        sd_name = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(sd_name, "reference_model.json"), "r") as f:
            json_reference = json.load(f)
        assert json_vars == json_reference
