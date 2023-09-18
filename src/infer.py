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

        if tag["start"] >= word["start"] and tag["end"] <= word["end"]:
            # tag is within the duration of a word
            # skip for now
            t += 1
        elif tag["start"] < word["start"]:
            # if the tag begins before the word
            result.append(tag)
            result.append(word)

            if tag["end"] <= word["end"]:
                # if the tag ends during the word
                t += 1
            w += 1
        elif tag["start"] > word["end"]:
            # if the tag starts after the word
            result.append(word)
            w += 1
        elif tag["start"] <= word["end"]:
            # if the tag starts during the word
            result.append(word)
            w += 1

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
