def recall_at_k(retrieved, expected_doc, expected_page, k):

    expected_page = int(expected_page)
    expected_doc = str(expected_doc).strip()

    for i in range(min(k, len(retrieved))):

        item = retrieved[i]

        doc = str(item.get("doc_id", "")).strip()
        page = int(item.get("page", -1))

        if doc == expected_doc and page == expected_page:
            return 1

    return 0