import json

def parse_generated_qns():
    """
    generated questions are split into a.txt and b.txt because >50mb & cannot push to git lol

    returns list of tuples containing
        json containing qa, context

    """

    def read_file(filename):
        with open(filename, encoding="iso-8859-1") as f:
            return f.read().split("\n")

    raw = read_file("generated_qns/a.txt") + read_file("generated_qns/b.txt")

    out = []
    for line in raw:
        try:
            qa, context = line.split("\t")
            out.append((json.loads(qa), context))
        except:pass

    return out


if __name__ == "__main__":
    qas = parse_generated_qns()
    print(len(qas))