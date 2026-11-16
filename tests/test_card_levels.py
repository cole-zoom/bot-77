from bot77.readers.levels import name_clusters


def test_a_card_always_beside_another_gets_its_own_name():
    # the bot's deck: Giant and Mini P.E.K.K.A in every entry; one opponent entry without the Giant, and hand reads
    # that missed the Giant now and then: the Giant's cluster counts Mini P.E.K.K.A at least as often as Giant
    decks = {("g%d" % i, "bottom"): {"Giant", "Mini P.E.K.K.A", "Arrows"} for i in range(9)}
    decks[("g9", "bottom")] = {"Mini P.E.K.K.A", "Arrows"}  # Giant never reached the hand
    decks[("o1", "top")] = {"Mini P.E.K.K.A", "Knight"}
    decks[("o2", "top")] = {"Arrows", "Hog Rider"}
    g = [("g%d" % i, "bottom") for i in range(10)]
    clusters = {0: g,  # the Giant's tiles (g9's hand read missed it)
                1: g + [("o1", "top")],  # Mini P.E.K.K.A's
                2: g + [("o2", "top")]}  # Arrows'
    names = {j: v[0] for j, v in name_clusters(clusters, decks).items()}
    assert names == {0: "Giant", 1: "Mini P.E.K.K.A", 2: "Arrows"}


def test_a_cluster_below_purity_stays_unnamed():
    decks = {("a", "bottom"): {"Giant"}, ("b", "bottom"): {"Knight"}}
    assert name_clusters({0: list(decks)}, decks) == {}
