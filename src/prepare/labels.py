"""Turn the NHTSA component field into one label per complaint."""
from src import config


def top_level_component(component):
    """'ENGINE AND ENGINE COOLING:ENGINE:GASOLINE' becomes 'ENGINE AND ENGINE COOLING'.

    Component names can contain commas ('SERVICE BRAKES, HYDRAULIC'), so we cut at the first colon only.
    """
    if not isinstance(component, str):
        return ""
    return component.split(":")[0].strip().upper()


def drop_multi_component(frame):
    """Remove complaints that list more than one top-level component.

    In the flat file a complaint with several components appears as several rows with the same
    ODI number. If those rows have different top-level components, the label is not clear, so the
    whole complaint is dropped. Rows that only differ in the sub-component are kept (once).

    Returns the cleaned frame and the number of complaints that were dropped.
    """
    distinct = frame.groupby("odi_no")["top_level"].nunique()
    multi = set(distinct[distinct > 1].index)
    single = frame[~frame["odi_no"].isin(multi)]
    single = single.drop_duplicates("odi_no", keep="first")
    return single, len(multi)


def most_frequent_classes(frame, top_k=config.TOP_K):
    """The top_k most frequent top-level components (not counting NHTSA's catch-all)."""
    counts = frame.loc[frame["top_level"] != config.NHTSA_CATCH_ALL, "top_level"].value_counts()
    return list(counts.index[:top_k])


def map_labels(frame, classes):
    """Add the label column: a frequent class keeps its name, everything else becomes OTHER."""
    frame = frame.copy()
    frame["label"] = frame["top_level"].where(frame["top_level"].isin(classes), config.OTHER)
    return frame
