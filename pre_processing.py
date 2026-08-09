from collections import Counter
from langdetect import detect
from datasketch import MinHash, MinHashLSH
import numpy as np

def preprocess(docs):

    def get_word_ngrams(text, n=5):

        words = text.lower().split()

        ngrams = []

        for i in range(len(words) - n + 1):

            piece = ""

            for j in range(n):
                if j == 0:
                    piece = words[i + j]
                else:
                    piece = piece + " " + words[i + j]

            ngrams.append(piece)

        return ngrams


    def deduplicate(docs, similarity_threshold=0.75, num_hashes=128):

        lsh = MinHashLSH(threshold=similarity_threshold, num_perm=num_hashes)

        kept = []

        for i in range(len(docs)):

            doc = docs[i]

            if len(doc.split()) < 10:
                kept.append(doc)

            else:

                signature = MinHash(num_perm=num_hashes)

                ngrams = get_word_ngrams(doc)

                for item in ngrams:
                    signature.update(item.encode("utf-8"))

                result = lsh.query(signature)

                if len(result) == 0:
                    lsh.insert("doc_" + str(i), signature)
                    kept.append(doc)

        return kept


    # language filter
    kept = []

    for doc in docs:
        if len(doc) < 50:
            kept.append(doc)
        else:
            try:
                language = detect(doc)
                if language == "en":
                    kept.append(doc)
            except:
                pass

    docs = kept


    # adult content filter
    adult_words = [
        "porn",
        "xxx",
        "nude",
        "naked",
        "escort",
        "adult content",
        "explicit",
        "nsfw"
    ]

    kept = []

    for doc in docs:

        found = False

        for word in adult_words:
            if word in doc.lower():
                found = True

        if found == False:
            kept.append(doc)

    docs = kept


    # deduplication
    docs = deduplicate(docs)


    # c4 filters
    cookie_phrases = [
        "cookie policy",
        "use of cookies",
        "uses cookies",
        "accept cookies",
        "cookie notice"
    ]

    kept = []

    for doc in docs:
        if "javascript" not in doc.lower():
            kept.append(doc)

    docs = kept


    kept = []

    for doc in docs:

        found = False

        for phrase in cookie_phrases:
            if phrase in doc.lower():
                found = True

        if found == False:
            kept.append(doc)

    docs = kept


    kept = []

    for doc in docs:
        if "lorem ipsum" not in doc.lower():
            kept.append(doc)

    docs = kept


    kept = []

    for doc in docs:
        if "{" not in doc:
            kept.append(doc)

    docs = kept


    kept = []

    for doc in docs:

        word_count = len(doc.split())

        if word_count >= 50 and word_count <= 100000:
            kept.append(doc)

    docs = kept


    # statistical filters
    sentence_endings = set(".!?\"'")

    kept = []

    for doc in docs:

        lines = []

        for line in doc.split("\n"):
            if line.strip() != "":
                lines.append(line.strip())

        if len(lines) > 0:

            count = 0

            for line in lines:
                if line[-1] in sentence_endings:
                    count = count + 1

            ratio = count / len(lines)

            if ratio >= 0.12:
                kept.append(doc)

    docs = kept


    kept = []

    for doc in docs:

        lines = []

        for line in doc.split("\n"):
            if line.strip() != "":
                lines.append(line.strip())

        if len(lines) > 0:

            total_chars = 0

            for line in lines:
                total_chars = total_chars + len(line)

            line_counts = Counter(lines)

            duplicate_chars = 0

            for line in line_counts:

                count = line_counts[line]

                if count > 1:
                    duplicate_chars = duplicate_chars + (len(line) * (count - 1))

            ratio = duplicate_chars / total_chars

            if ratio <= 0.10:
                kept.append(doc)

    docs = kept


    kept = []

    for doc in docs:

        lines = []

        for line in doc.split("\n"):
            if line.strip() != "":
                lines.append(line.strip())

        if len(lines) > 0:

            short_lines = 0

            for line in lines:
                if len(line) < 30:
                    short_lines = short_lines + 1

            ratio = short_lines / len(lines)

            if ratio <= 0.67:
                kept.append(doc)

    docs = kept

    return docs