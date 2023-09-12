def combine(transcription, tags):
    result = []
    for info in tags:
        if len(info["audio tags"]) > 0:
            # assuming the first tag is most probable
            tag = info["audio tags"][0][0]
            time = info["time"]

            segments = transcription["segments"]
            # TODO: could cache a lot of this rather than iterating the whole thing
            for c, n in zip(segments[:-1], segments[1:]):
                if c["start"] <= time["start"] <= c["end"]:
                    # TODO: what if the audio tag spans multiple words, need to rework
                    if n["start"] <= time["end"] <= n["end"]:
                        # in this case, the audio tag has to go between both labels
                        # TODO: add more info
                        result.append(c["word"])
                        result.append(tag)
                        result.append(n["word"])
    return result
