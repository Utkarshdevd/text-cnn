# -*- coding: utf-8 -*-
import pandas
import json

generated_data = []
data = []

binaryClass = True


class AssembleDataset():
    def __init__(self, json_unit, class_type):
        if binaryClass and class_type is "jokes":
            self.label = 1
        elif binaryClass and class_type is "notJokes":
            self.label = 0
        else:
            self.label = -1
        self.concat_str = json_unit["concat"] if not pandas.isnull(json_unit["concat"]) else None
        self.score_term = json_unit["scoreTerm"] if not pandas.isnull(json_unit["scoreTerm"]) else None
        self.text_terms = json_unit["textTerm"].split(',') if not pandas.isnull(json_unit["textTerm"]) else None
        self.type = json_unit["type"]
        self.filePath = "{}/{}.{}".format(json_unit["path"], json_unit["filename"], self.type)

    def loadFile(self):
        if self.type == "json":
            with open(self.filePath) as file:
                _ = json.load(file)
                self.data = pandas.DataFrame.from_dict(_)
        elif self.type == "pickle":
            self.data = pandas.read_pickle(self.filePath)
        else:
            self.data = pandas.read_csv(self.filePath, header=0)

    def assemble(self):
        assembled_data = []
        if isinstance(self.data, pandas.DataFrame):
            print(self.filePath, self.data.shape)
            # print(self.data[self.score_term])
            for _, row in self.data.iterrows():
                text = None
                if self.text_terms is not None:
                    for key in self.text_terms:
                        if text is None:
                            text = row[key]
                        text = "{}{}{}".format(text, self.concat_str, row[key])
                if not binaryClass:
                    # TODO: Add rescaled score instead of binary class
                    pass
                assembled_data.append([self.label, text])
        else:
            print(self.filePath, len(self.data))
            for line in self.data:
                assembled_data.append([self.label, line])
            # print(self.data)
        return assembled_data


def runAssemble(json_spec, class_type):
    print(json_spec)
    full_data = []
    for _, joke in json_spec.iterrows():
        dset = AssembleDataset(joke, class_type)
        dset.loadFile()
        new_data = dset.assemble()
        full_data.extend(new_data)
    return full_data


def main():
    with open("data/datasetSpec.json") as dset_spec:
        data = json.load(dset_spec)
        json_spec = pandas.DataFrame.from_dict(data.get("jokes"))
        all_jokes = runAssemble(json_spec, "jokes")
        json_spec = pandas.DataFrame.from_dict(data.get("notJokes"))
        all_not_jokes = runAssemble(json_spec, "notJokes")
        print(len(all_jokes), len(all_not_jokes))
        all_jokes.extend(all_not_jokes)
        print(len(all_jokes))
        save = pandas.DataFrame.from_records(all_jokes)
        save.to_csv("data/generated_data/binary_set.csv", index=False)


if __name__ == "__main__":
    main()
