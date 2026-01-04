
<br>


> [!WARNING]
> 
> - Due to free quota limits, embedding model can save upto 100 chunks only
> - It's glue code is written in `faiss_manager.py`

<br>
<br>

## Problems and their Solutions


### 1) Page number disorder while loading chapter content, input of unwanted book pages.

Add a node in `ChapterContentLoader` subagent that tells the page number of first chapter
    and according to that exact content will be load.



### 2) Huge time while fetching a particular chapter.

Reason: Most use of Glue code
    Solution: Use genunine methods to load a particular chapter content. Handle this situation on `PDFProcessor().load_pdf()` in `pdf_processor.py`





