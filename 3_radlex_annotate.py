import json
import os
import threading
from collections import defaultdict

import pandas
import spacy
from spacy.language import Language
from spacy.matcher import Matcher
from spacy.tokens import Doc
from tqdm import tqdm


class OntologyNode:
    def __init__(self, row_idx, class_id, class_name, df_row):
        self.row_idx = row_idx
        self.class_id = class_id
        self.class_name = class_name
        self.synonyms = [] if df_row["Synonyms"] == "" else df_row["Synonyms"].split("|")
        self.df_row = df_row

        # The tree structure is maintained by the parent and children attributes. Only one level of parent-child relationship is maintained.
        self.parent = []
        self.children = []
        self.is_root = False
        self.tree_level = None

        # It's parents from all levels
        self._all_parents = []

    def add_child(self, child):
        self.children.append(child)

    def add_parent(self, parent):
        self.parent.append(parent)

    @property
    def all_parents(self):
        if self.is_root:
            return []
        elif self._all_parents:
            return self._all_parents
        else:
            for parent in self.parent:
                # 避免父节点重复
                self._all_parents = set(parent.all_parents + [parent])
                self._all_parents = list(self._all_parents)
            return self._all_parents

    def __eq__(self, other):
        if isinstance(other, OntologyNode):
            return self.class_id == other.class_id
        else:
            return self.class_id == other

    def __hash__(self):
        return hash(self.class_id)

    def __str__(self):
        return f"{self.class_id}: {self.class_name}"

    def __repr__(self):
        return self.__str__()


def set_tree_level(curr_node, tree_level):
    curr_node.tree_level = tree_level
    for child in curr_node.children:
        set_tree_level(child, tree_level + 1)
    if not curr_node.children:
        return


def build_radlex_tree(df_csv):
    # Build a RadLex node list
    node_list = []
    root_node = None
    for idx, row in tqdm(df_csv.iterrows(), total=df_csv.shape[0], desc="Building RadLex tree"):
        ontology_node = OntologyNode(row_idx=idx, class_id=row["Class ID"], class_name=row["Preferred Label"], df_row=row)
        if row["Preferred Label"] in row["Class ID"]:
            ontology_node.class_name = row["http://radlex.org/RID/Preferred_Name_for_Obsolete"]
        node_list.append(ontology_node)

    # Resolve the node list and build a RadLex tree
    for node in tqdm(node_list, total=len(node_list), desc="Building RadLex tree"):
        df_row = node.df_row
        parent_ids = df_row["Parents"].split("|")
        for parent_id in parent_ids:
            parent_row_indices = df_csv.loc[df_csv["Class ID"] == parent_id].index
            if not parent_row_indices.empty:
                parent_row_idx = parent_row_indices[0]
                parent_node = node_list[parent_row_idx]
                assert parent_node.class_id == parent_id
                node.add_parent(parent_node)
                parent_node.add_child(node)
            else:
                # In radlex, http://radlex.org/RID/RID0 has parent http://www.w3.org/2002/07/owl#Thing.
                # However, the RID0 is already the root node in the RadLex ontology. We can safely ignore the owl#Thing.
                root_node = node
                node.is_root = True
                node.tree_level = 0

    return node_list, root_node


def fill_matcher(matcher, radlex_nodes):
    radlex_id2notPunctToks_dict = defaultdict(list)

    no_lemma_match_toks = ["left", "bound", "axes", "wound", "saw"]

    for node in tqdm(radlex_nodes, desc="Filling matcher"):
        # Chemical elements requires different matching patterns
        is_chemical_element = True if "http://radlex.org/RID/RID11756" in node.all_parents else False
        if is_chemical_element:
            pattern = [{"LOWER": tok.text.lower()} for tok in nlp(node.class_name)]
            matcher.add(f"{node.class_id}@text", [pattern])
            # e.g. Ho (holmium), Ne (neon), B (boron), I (iodine), H (hydrogen), C (carbon), N (nitrogen), etc.
            for term in node.synonyms:
                # Omit single character synonyms to avoid confusion with other words.
                if not len(term) == 1:
                    pattern = [{"TEXT": tok.text} for tok in nlp(term)]
                    matcher.add(f"{node.class_id}@text", [pattern])
        else:
            terms = [node.class_name] + node.synonyms
            for idx, term in enumerate(terms):
                doc = nlp(term)

                # Exact matching under lower case
                pattern = [{"LOWER": tok.text.lower()} for tok in doc]
                matcher.add(f"{node.class_id}@lower_text", [pattern])

                # More general matching patterns
                pattern = []
                for tok in doc:
                    if tok.is_punct:
                        pattern.append({"IS_PUNCT": True, "OP": "*"})  # Zero or more punctuations
                    elif tok.is_upper:
                        pattern.append({"LOWER": tok.text.lower()})  # Abbreviation such as MS (multiple sclerosis),
                    elif tok.text in no_lemma_match_toks:
                        pattern.append({"LOWER": tok.text.lower()})
                    else:
                        pattern.append({"LEMMA": tok.lemma_})
                matcher.add(f"{node.class_id}@lemma", [pattern])  # If a pattern already exists for the given ID, the patterns will be extended. An on_match callback will be overwritten.

                # Fuzzy matching
                pattern = []
                if len(doc) > 1:
                    # Example: f-c, follow-up
                    for tok in doc:
                        if tok.is_punct:
                            pattern.append({"IS_PUNCT": True, "OP": "*"})  # Zero or more punctuations
                        else:
                            if tok.is_upper or tok.text in no_lemma_match_toks:
                                pattern.append({"LOWER": tok.text.lower()})  # Abbreviation do not use FUZZY, such as MS (multiple sclerosis),
                            else:
                                if len(tok.text) >= 16:
                                    pattern.append({"LEMMA": {"FUZZY3": tok.lemma_}})
                                elif len(tok.text) >= 12:
                                    pattern.append({"LEMMA": {"FUZZY2": tok.lemma_}})
                                elif len(tok.text) >= 8:
                                    pattern.append({"LEMMA": {"FUZZY1": tok.lemma_}})
                                else:
                                    # len(tok.text) < 8
                                    # For multi-token term, do not use FUZZY if the length of the token is less than 8.
                                    pattern.append({"LEMMA": tok.lemma_})

                            # For fuzzy_lemma, we need to append the idx to the key to distinguish different terms (sysnonyms),
                            # as we need to compare the first character later in the fuzzy_match_filter()
                            radlex_id2notPunctToks_dict[f"{node.class_id}@fuzzy_lemma#{idx}"].append(tok.text)
                if len(doc) == 1:
                    tok = doc[0]
                    # Not an abbreviation, not in the no_lemma_match_toks list
                    if not tok.is_upper and not tok.text in no_lemma_match_toks:
                        if len(tok.text) >= 16:
                            pattern.append({"LEMMA": {"FUZZY3": tok.lemma_}})
                        elif len(tok.text) >= 12:
                            pattern.append({"LEMMA": {"FUZZY2": tok.lemma_}})
                        elif len(tok.text) >= 8:
                            pattern.append({"LEMMA": {"FUZZY1": tok.lemma_}})

                    radlex_id2notPunctToks_dict[f"{node.class_id}@fuzzy_lemma#{idx}"].append(tok.text)

                def fuzzy_match_filter(matcher, input_doc, i, matches):
                    match_id, start, end = matches[i]
                    radlex_id = nlp.vocab.strings[match_id]
                    term_toks = radlex_id2notPunctToks_dict[radlex_id]
                    span = input_doc[start:end]  # The matched span

                    # print("on_match_func", nlp.vocab.strings[match_id], "text span:", span, ", term toks:", term_toks)
                    for text_tok, term_tok_str in zip([tok for tok in span if not tok.is_punct], [tok for tok in term_toks]):
                        # If the fuzzy matching do not match on the first character, the match is considered invalid.
                        if text_tok.text.lower()[0] != term_tok_str.lower()[0]:
                            matches[i] = (None, None, None)
                            break

                if pattern:
                    matcher.add(f"{node.class_id}@fuzzy_lemma#{idx}", [pattern], on_match=fuzzy_match_filter)

    return matcher


def set_match_component(nlp, matcher, radlex_id2name_dict):
    Doc.set_extension("matched_radlex_dict", default=dict(), force=True)

    @Language.component("match_component")
    def radlex_matcher(doc):
        matches = matcher(doc)
        for match_id, start, end in matches:
            if match_id is None:
                continue

            span = doc[start:end]  # The matched span
            radlex_id = nlp.vocab.strings[match_id]  # Get string representation

            # e.g
            # http://radlex.org/RID/RID5978@lower_text
            # http://radlex.org/RID/RID5978@lemma
            # http://radlex.org/RID/RID5978@fuzzy_lemma#0
            radlex_id, match_type = radlex_id.split("@")
            match_type = match_type.split("#")[0]

            # The matching order is alwarys: text match -> lower_text match -> lemma match -> fuzzy_lemma match
            # We only keep the first matched result for each matched span. e.g. if text matched, we skip the reast of the matches.
            unique_id = f"{radlex_id}@{start}|{end}"
            if unique_id not in doc._.matched_radlex_dict:
                doc._.matched_radlex_dict[unique_id] = {
                    "match_type": match_type,
                    "radlex_id": radlex_id,
                    "radlex_name": radlex_id2name_dict[radlex_id],
                    "matched_text": span.text,
                    "char_indices": (span.start_char, span.end_char),
                    "tok_indices": (start, end),
                }
        return doc

    nlp.add_pipe("match_component", name="radlex_matcher", last=True)
    print(nlp.pipe_names)


def count_file_lines(file_path):
    with open(file_path, "rb") as f:
        return sum(1 for _ in f)


def read_large_file(input_path):
    with open(input_path, "r") as f:
        for line in f:
            doc = json.loads(line.strip())
            sent_text = doc["split_sent_text"].strip()
            assert sent_text == sent_text.strip(), doc
            yield (sent_text, {"doc_key": doc["doc_key"]})


if __name__ == "__main__":
    radlex_csv_path = "/home/yuxiang/liao/resources/bioportal/radlex/RADLEX.csv"
    df_radlex_csv = pandas.read_csv(radlex_csv_path, keep_default_na=False)
    radlex_nodes, radlex_root_node = build_radlex_tree(df_radlex_csv)
    radlex_id2name_dict = {node.class_id: node.class_name for node in radlex_nodes}
    print(f"Number of RadLex nodes: {len(radlex_nodes)}")

    # Tracing all parents of nodes
    for node in radlex_nodes:
        node.all_parents

    set_tree_level(radlex_root_node, tree_level=0)

    nlp = spacy.load("en_core_web_sm", exclude=["ner"])
    matcher = Matcher(nlp.vocab)
    fill_matcher(matcher, radlex_nodes)
    set_match_component(nlp, matcher, radlex_id2name_dict)

    input_path = "/home/yuxiang/liao/workspace/arrg_preprocessing/outputs/interpret_sents/raw/raw_sents.json"

    output_file_path = "/home/yuxiang/liao/workspace/arrg_preprocessing/outputs/interpret_sents/radlex_annotate/radlex_ann.json"
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    f = open(output_file_path, "w", encoding="utf-8")

    doc_tuples = nlp.pipe(read_large_file(input_path), as_tuples=True, n_process=8)
    for doc, info_dict in tqdm(doc_tuples, total=count_file_lines(input_path)):
        output_dict = {"doc_key": info_dict["doc_key"], "sent_text": doc.text, "radlex": []}

        for _, match_dict in doc._.matched_radlex_dict.items():
            output_dict["radlex"].append(match_dict)

        f.write(json.dumps(output_dict))
        f.write("\n")
        f.flush()

    f.close()
