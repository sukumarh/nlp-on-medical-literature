import numpy as np
from sklearn.metrics.pairwise import cosine_similarity as cos_sim
from scipy.stats import pearsonr as correlation
from multiprocessing import Pool


class WordEmbeddings:

    def __init__(self, gloveFileName):
        self.glove = {}
        self.__loadGlove(gloveFileName)

    def __loadGlove(self, glove_f):
        print("Loading glove")
        with open(glove_f, encoding="utf8") as glove_lines:
            for glove_line in glove_lines:
                if glove_line[0] == "#":
                    continue
                key, *value = glove_line.rstrip().split(' ')
                v = list(map(float, value))
                self.glove[key] = v

        if len(self.glove) > 0:
            print("Glove loaded successfully.")

    def evaluateSimilarity(self, corpusFileName, outputFileName):
        print("Evaluating Word Similarity")
        machine_scores = []
        human_scores = []
        not_found = 0
        words_not_found = []

        output = open(outputFileName, 'w')
        output.write("# Word 1\tWord 2\tHuman (mean)\tMachine\n")

        with open(corpusFileName) as corpus_lines:
            for corpus_line in corpus_lines:
                if corpus_line[0] == "#":
                    continue

                # Reading word from the corpus
                line = {}
                line['tag'], line['word_1'], line['word_2'], line['human_score'] = corpus_line.rstrip().split('\t')

                # Retrieving the vectors of the words
                if line['word_1'] not in self.glove:
                    not_found += 1
                    words_not_found.append(line['word_1'])
                    continue

                if line['word_2'] not in self.glove:
                    not_found += 1
                    words_not_found.append(line['word_2'])
                    continue

                word1_vec = np.array(self.glove[line['word_1']])
                word2_vec = np.array(self.glove[line['word_2']])

                # Computing the score based on the two vectors
                machine_score = cos_sim(word1_vec.reshape(1, -1), word2_vec.reshape(1, -1))[0][0] * 10

                machine_scores.append(machine_score)

                # Human score
                human_scores.append(float(line['human_score']))

                # the pair, the human score, and the word embeddings score, and the overall correlation.
                o = '\t'.join([line['tag'], line['word_1'], line['word_2'], line['human_score'],
                               str(round(machine_score, 4))])
                output.write(o + '\n')

        # Evaluate score - compute correlation of the two scores
        evaluation = correlation(human_scores, machine_scores)
        evaluation = round(evaluation[0], 4)
        output.write("# Correlation = " + str(evaluation) + "\n")
        output.close()
        print("Evaluation complete.")
        return evaluation

    def __getVectors(self, x):
        z = np.array(self.glove[x]).reshape(1, -1)
        return z

    def analogy(self, analogies, outputFileName):
        for a, b, c in analogies:
            print("Analogy to find: " + a + ":" + b + "::" + c + ": ?")

        output = open(outputFileName, mode="a+")
        output.write("# Analogies\n")

        p = Pool(len(analogies))
        d_ = p.map(self.thread, analogies)

        for [a, b, c], d in zip(analogies, d_):
            output.write(a + ":" + b + "::" + c + ":" + d[0] + "\n")
            print("Analogy found: " + a + ":" + b + "::" + c + ":" + d[0])

    def thread(self, analogy):
        a, b, c = analogy
        a_, b_, c_ = self.__getVectors(a), self.__getVectors(b), self.__getVectors(c)
        d_ = b_ - a_ + c_
        d = ""
        max_score = 0
        for i in self.glove:
            if i == a or i == b or i == c:
                continue
            score = cos_sim(d_, np.array(self.glove[i]).reshape(1, -1))[0][0] * 10
            if score > max_score:
                max_score = score
                d = i
        return d, max_score


def main():
    # File Names
    corpusFileName = "wordsim-353.txt"
    gloveFileName = "glove.6B.300d.txt"
    outputFileName = "output.txt"

    we = WordEmbeddings(gloveFileName)

    evaluation = we.evaluateSimilarity(corpusFileName, outputFileName)

    print("Word Similarity evaluation is " + str(evaluation))

    analogies = [
        ['king', 'man', 'queen'],
        ['in', 'out', 'up'],
        ['doctor', 'hospital', 'teacher'],
        ['author', 'story', 'poet']
    ]

    we.analogy(analogies, outputFileName)


if __name__ == '__main__':
    main()
