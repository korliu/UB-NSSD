from time import sleep


def combine(transcription, audio_tags):
    return _combine(
        words(transcription),
        tags(audio_tags),
    )


def _combine(words, tags):
    # little preallocation
    # result = [None] * (len(words) + len(tags))
    result = []

    # some variation of a merge sort
    w, t = 0, 0
    while w < len(words) and t < len(tags):
        word = words[w]
        tag = tags[t]

        # NOTE: for testing
        # print(word, tag)
        # print()
        # sleep(0.1)

        if tag["start"] < word["start"]:
            result.append(tag)
            t += 1
        else:
            result.append(word)
            w += 1

    for i in range(w, len(words)):
        result.append(words[i])

    for i in range(t, len(tags)):
        result.append(tags[i])

    # TODO: temp, super inefficient impl to dedup
    for i in range(len(result) - 1, 1, -1):
        if "tag" in result[i - 1] and "tag" in result[i]:
            if result[i - 1]["tag"] == result[i]["tag"]:
                result.pop(i)

    return result


def tags(audio_tags):
    tags = []
    for info in audio_tags:
        found = info["audio tags"]
        # if there is an audio tag for this segment
        if len(found) > 0:
            tags.append(
                {
                    # assume the first audio tag is the most probable
                    "tag": found[0][0],
                    "start": info["time"]["start"],
                    "end": info["time"]["end"],
                }
            )

    return tags


def words(transcription):
    words = []
    for segment in transcription["segments"]:
        for info in segment["words"]:
            words.append(
                {
                    "word": info["word"],
                    "start": info["start"],
                    "end": info["end"],
                }
            )
    return words
