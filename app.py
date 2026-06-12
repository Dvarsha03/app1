import streamlit as st
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import HttpResponseError
from PIL import Image, ImageDraw
import io
import json
import requests

st.set_page_config(
    page_title="Azure CV Toolkit",
    page_icon="👁",
    layout="wide"
)

# ══════════════════════════════════════════════════════════════
# SIDEBAR — CREDENTIALS
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.title("Credentials")
    st.caption("Enter your Azure credentials to unlock features")

    st.markdown("### Computer Vision")
    cv_endpoint = st.text_input(
        "CV Endpoint",
        placeholder="https://your-resource.cognitiveservices.azure.com/",
        type="default"
    )
    cv_key = st.text_input(
        "CV Key",
        placeholder="Enter your CV Key 1",
        type="password"
    )

    st.divider()

    st.markdown("### Document Intelligence")
    di_endpoint = st.text_input(
        "DI Endpoint",
        placeholder="https://your-resource.cognitiveservices.azure.com/",
        type="default"
    )
    di_key = st.text_input(
        "DI Key",
        placeholder="Enter your DI Key 1",
        type="password"
    )

    st.divider()

    st.markdown("### Status")
    cv_ready = bool(cv_endpoint and cv_key)
    di_ready = bool(di_endpoint and di_key)

    if cv_ready:
        st.success("Computer Vision — ready")
    else:
        st.warning("Computer Vision — not entered")

    if di_ready:
        st.success("Document Intelligence — ready")
    else:
        st.warning("Document Intelligence — not entered")

# ══════════════════════════════════════════════════════════════
# CREATE CLIENTS
# ══════════════════════════════════════════════════════════════
cv_client = None
di_client = None

if cv_ready:
    try:
        cv_client = ImageAnalysisClient(
            endpoint=cv_endpoint,
            credential=AzureKeyCredential(cv_key)
        )
    except Exception:
        st.sidebar.error("CV credentials invalid")
        cv_client = None
        cv_ready  = False

if di_ready:
    try:
        di_client = DocumentIntelligenceClient(
            endpoint=di_endpoint,
            credential=AzureKeyCredential(di_key)
        )
    except Exception:
        st.sidebar.error("DI credentials invalid")
        di_client = None
        di_ready  = False

# ══════════════════════════════════════════════════════════════
# MAIN HEADER
# ══════════════════════════════════════════════════════════════
st.title("Azure Computer Vision Toolkit")
st.caption(
    "Image Analysis · OCR · Object Detection · "
    "Dense Captions · People Detection · Document Intelligence"
)

# ══════════════════════════════════════════════════════════════
# OVERVIEW — always visible to everyone
# ══════════════════════════════════════════════════════════════
with st.expander("Application overview — what this app does", expanded=True):
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### Image Vision")
        st.markdown("""
- **Caption** — one sentence describing the image
- **Tags** — keywords found in the image
- **Object detection** — bounding boxes around objects
- **Dense captions** — caption for every region
- **People detection** — find and locate people
- **OCR** — extract all text from image
- Supports single or multiple images
- Upload from PC or enter URL
        """)

    with col2:
        st.markdown("#### Document Reader")
        st.markdown("""
- **Receipt reader** — merchant, date, total, items
- **Invoice reader** — vendor, ID, dates, line items
- Single or multiple documents
- Upload from PC or enter URL manually
- Download results as JSON
        """)

    with col3:
        st.markdown("#### How to use")
        st.markdown("""
1. Enter credentials in the sidebar
2. Choose Image Vision or Document Reader tab
3. Upload image(s) from PC or enter URL(s)
4. Choose detect all or specific feature
5. Click analyse / read
6. View results and download JSON
        """)

st.divider()

# ══════════════════════════════════════════════════════════════
# MAIN TABS
# ══════════════════════════════════════════════════════════════
tab_vision, tab_docs = st.tabs(["Image Vision", "Document Reader"])

# ══════════════════════════════════════════════════════════════
# TAB 1 — IMAGE VISION
# ══════════════════════════════════════════════════════════════
with tab_vision:

    if not cv_ready:
        st.warning("Please enter your Computer Vision endpoint and key in the sidebar.")
        st.info("Create a Computer Vision resource at portal.azure.com in East US region.")
    else:
        st.subheader("Image Vision")

        # Step 1
        st.markdown("#### Step 1 — Choose image source")
        image_source = st.radio(
            "Image source",
            ["Upload from PC", "Enter image URL(s)"],
            horizontal=True
        )

        images_to_process = []

        if image_source == "Upload from PC":
            img_count = st.radio(
                "How many images?",
                ["Single image", "Multiple images"],
                horizontal=True
            )
            if img_count == "Single image":
                uploaded = st.file_uploader(
                    "Upload one image",
                    type=["jpg", "jpeg", "png", "bmp"]
                )
                if uploaded:
                    images_to_process = [("file", uploaded, uploaded.name)]
            else:
                uploaded_files = st.file_uploader(
                    "Upload multiple images",
                    type=["jpg", "jpeg", "png", "bmp"],
                    accept_multiple_files=True
                )
                if uploaded_files:
                    images_to_process = [
                        ("file", f, f.name) for f in uploaded_files
                    ]
        else:
            url_count = st.radio(
                "How many URLs?",
                ["Single URL", "Multiple URLs"],
                horizontal=True
            )
            if url_count == "Single URL":
                url = st.text_input(
                    "Image URL",
                    placeholder="https://example.com/image.jpg"
                )
                if url:
                    images_to_process = [("url", url, url.split("/")[-1])]
            else:
                urls_text = st.text_area(
                    "Enter one URL per line",
                    placeholder=(
                        "https://example.com/image1.jpg\n"
                        "https://example.com/image2.jpg"
                    ),
                    height=120
                )
                if urls_text:
                    images_to_process = [
                        ("url", u.strip(), u.strip().split("/")[-1])
                        for u in urls_text.strip().split("\n")
                        if u.strip()
                    ]

            with st.expander("Need sample image URLs to test?"):
                st.code("https://images.unsplash.com/photo-1587300003388-59208cc962cb?w=800")
                st.caption("Labrador dog — good for caption, tags, object detection")
                st.code("https://images.unsplash.com/photo-1529156069898-49953e39b3ac?w=800")
                st.caption("Street scene — good for people detection, dense captions")

        if images_to_process:
            st.markdown(f"**{len(images_to_process)} image(s) ready to process**")

        # Step 2
        st.markdown("#### Step 2 — Choose what to detect")
        detect_mode = st.radio(
            "Detection mode",
            ["Detect all", "Detect specific"],
            horizontal=True
        )

        if detect_mode == "Detect all":
            selected_features = [
                "Caption", "Tags", "Object detection",
                "Dense captions", "People detection", "OCR"
            ]
            st.info("All 6 features will run on your image(s)")
        else:
            selected_features = st.multiselect(
                "Select features to detect",
                [
                    "Caption", "Tags", "Object detection",
                    "Dense captions", "People detection", "OCR"
                ],
                default=["Caption", "Tags"]
            )

        feature_map = {
            "Caption":          VisualFeatures.CAPTION,
            "Tags":             VisualFeatures.TAGS,
            "Object detection": VisualFeatures.OBJECTS,
            "Dense captions":   VisualFeatures.DENSE_CAPTIONS,
            "People detection": VisualFeatures.PEOPLE,
            "OCR":              VisualFeatures.READ
        }

        visual_features = [
            feature_map[f] for f in selected_features if f in feature_map
        ]

        # Step 3
        st.markdown("#### Step 3 — Run analysis")

        if st.button("Analyse", key="btn_analyse", type="primary"):
            if not images_to_process:
                st.warning("Please upload at least one image or enter a URL.")
            elif not selected_features:
                st.warning("Please select at least one feature to detect.")
            else:
                for idx, (src_type, img_source, img_name) in enumerate(images_to_process):
                    st.markdown("---")
                    st.markdown(f"#### Image {idx+1} — {img_name}")

                    try:
                        if src_type == "file":
                            img_pil = Image.open(img_source).convert("RGB")
                        else:
                            r = requests.get(img_source, timeout=15)
                            img_pil = Image.open(io.BytesIO(r.content)).convert("RGB")

                        buf = io.BytesIO()
                        img_pil.save(buf, format="JPEG", quality=95)
                        image_bytes = buf.getvalue()

                    except Exception as e:
                        st.error(f"Could not load image: {e}")
                        continue

                    st.image(img_pil, width=400, caption=f"Image {idx+1}")

                    with st.spinner(f"Analysing image {idx+1}..."):
                        try:
                            result = cv_client.analyze(
                                image_data=image_bytes,
                                visual_features=visual_features,
                                language="en"
                            )
                        except HttpResponseError as e:
                            st.error(f"Azure CV error: {e.message}")
                            continue

                    all_results = {}

                    # Caption
                    if "Caption" in selected_features and result.caption:
                        st.markdown("**Caption**")
                        st.success(result.caption.text)
                        st.metric("Confidence", f"{result.caption.confidence*100:.1f}%")
                        all_results["caption"] = {
                            "text": result.caption.text,
                            "confidence": round(result.caption.confidence, 4)
                        }

                    # Tags
                    if "Tags" in selected_features and result.tags:
                        st.markdown("**Tags**")
                        cols = st.columns(4)
                        tags_data = []
                        for i, tag in enumerate(result.tags.list):
                            with cols[i % 4]:
                                emoji = "🟢" if tag.confidence > 0.9 else "🟡" if tag.confidence > 0.7 else "🔴"
                                st.markdown(f"{emoji} **{tag.name}**  \n{tag.confidence*100:.0f}%")
                            tags_data.append({"name": tag.name, "confidence": round(tag.confidence, 4)})
                        all_results["tags"] = tags_data

                    # Object Detection
                    if "Object detection" in selected_features:
                        st.markdown("**Object Detection**")
                        if result.objects and result.objects.list:
                            img_draw = img_pil.copy()
                            draw     = ImageDraw.Draw(img_draw)
                            colors   = ["red", "blue", "green", "orange", "purple", "yellow"]
                            objects_data = []
                            for i, obj in enumerate(result.objects.list):
                                b     = obj.bounding_box
                                name  = obj.tags[0].name
                                conf  = obj.tags[0].confidence
                                color = colors[i % len(colors)]
                                draw.rectangle([b.x, b.y, b.x+b.width, b.y+b.height], outline=color, width=4)
                                draw.rectangle([b.x, b.y-25, b.x+len(name)*9+65, b.y], fill=color)
                                draw.text((b.x+5, b.y-20), f"{name} {conf*100:.0f}%", fill="white")
                                objects_data.append({
                                    "name": name, "confidence": round(conf, 4),
                                    "bounding_box": {"x": b.x, "y": b.y, "width": b.width, "height": b.height}
                                })
                            col1, col2 = st.columns([2, 1])
                            with col1:
                                st.image(img_draw, width=400)
                            with col2:
                                st.markdown(f"**{len(objects_data)} objects found:**")
                                for o in objects_data:
                                    st.markdown(f"- **{o['name']}** ({o['confidence']*100:.0f}%)")
                            all_results["objects"] = objects_data
                        else:
                            st.info("No objects detected")

                    # Dense Captions
                    if "Dense captions" in selected_features:
                        st.markdown("**Dense Captions**")
                        if result.dense_captions and result.dense_captions.list:
                            img_draw  = img_pil.copy()
                            draw      = ImageDraw.Draw(img_draw)
                            caps_data = []
                            for i, cap in enumerate(result.dense_captions.list):
                                b = cap.bounding_box
                                draw.rectangle([b.x, b.y, b.x+b.width, b.y+b.height], outline="cyan", width=2)
                                draw.text((b.x+4, b.y+4), str(i+1), fill="cyan")
                                caps_data.append({
                                    "region": i+1, "caption": cap.text,
                                    "confidence": round(cap.confidence, 4)
                                })
                            col1, col2 = st.columns([2, 1])
                            with col1:
                                st.image(img_draw, width=400)
                            with col2:
                                for c in caps_data:
                                    st.markdown(f"**{c['region']}.** {c['caption']}  \nconf: {c['confidence']*100:.0f}%")
                            all_results["dense_captions"] = caps_data

                    # People Detection
                    if "People detection" in selected_features:
                        st.markdown("**People Detection**")
                        if result.people and result.people.list:
                            detected = [p for p in result.people.list if p.confidence > 0.5]
                            if detected:
                                st.success(f"{len(detected)} person(s) detected")
                                img_draw    = img_pil.copy()
                                draw        = ImageDraw.Draw(img_draw)
                                people_data = []
                                for i, person in enumerate(detected):
                                    b = person.bounding_box
                                    draw.rectangle([b.x, b.y, b.x+b.width, b.y+b.height], outline="lime", width=3)
                                    draw.rectangle([b.x, b.y-22, b.x+130, b.y], fill="lime")
                                    draw.text((b.x+4, b.y-18), f"Person {i+1} {person.confidence*100:.0f}%", fill="black")
                                    people_data.append({"person": i+1, "confidence": round(person.confidence, 4)})
                                col1, col2 = st.columns([2, 1])
                                with col1:
                                    st.image(img_draw, width=400)
                                with col2:
                                    for p in people_data:
                                        st.metric(f"Person {p['person']}", f"{p['confidence']*100:.0f}%")
                                all_results["people"] = people_data
                            else:
                                st.info("No people detected with confidence > 50%")
                        else:
                            st.info("No people detected")

                    # OCR
                    if "OCR" in selected_features:
                        st.markdown("**OCR — Text Extracted**")
                        if result.read and result.read.blocks:
                            all_lines = []
                            for block in result.read.blocks:
                                for line in block.lines:
                                    st.markdown(f"`{line.text}`")
                                    all_lines.append(line.text)
                            full_text = "\n".join(all_lines)
                            st.text_area("Full extracted text", full_text, height=100)
                            all_results["ocr_text"] = all_lines
                        else:
                            st.info("No text detected in this image")

                    # Download
                    if all_results:
                        st.download_button(
                            f"Download results — image {idx+1} JSON",
                            data=json.dumps(all_results, indent=2),
                            file_name=f"results_image_{idx+1}.json",
                            mime="application/json"
                        )


# ══════════════════════════════════════════════════════════════
# TAB 2 — DOCUMENT READER
# ══════════════════════════════════════════════════════════════
with tab_docs:

    if not di_ready:
        st.warning("Please enter your Document Intelligence endpoint and key in the sidebar.")
        st.info("Create a Document Intelligence resource at portal.azure.com. Free tier (F0) allows 500 pages per month.")
    else:
        st.subheader("Document Reader")

        # Step 1
        st.markdown("#### Step 1 — Choose document type")
        doc_type = st.radio(
            "What do you want to read?",
            ["Receipt only", "Invoice only", "Both receipt and invoice"],
            horizontal=True
        )

        # Step 2
        st.markdown("#### Step 2 — Single or multiple documents")
        doc_count = st.radio(
            "How many documents?",
            ["Single document", "Multiple documents"],
            horizontal=True
        )

        # Step 3
        st.markdown("#### Step 3 — Choose document source")
        doc_input_method = st.radio(
            "How do you want to provide the document?",
            ["Upload from PC", "Enter URL manually"],
            horizontal=True
        )

        receipt_urls  = []
        invoice_urls  = []
        receipt_files = []
        invoice_files = []

        if doc_input_method == "Upload from PC":

            if doc_type in ["Receipt only", "Both receipt and invoice"]:
                st.markdown("**Upload Receipt(s)**")
                if doc_count == "Single document":
                    r_file = st.file_uploader(
                        "Upload receipt",
                        type=["jpg", "jpeg", "png", "pdf", "bmp"],
                        key="receipt_single"
                    )
                    if r_file:
                        receipt_files = [r_file]
                else:
                    r_files = st.file_uploader(
                        "Upload receipts",
                        type=["jpg", "jpeg", "png", "pdf", "bmp"],
                        accept_multiple_files=True,
                        key="receipt_multi"
                    )
                    if r_files:
                        receipt_files = r_files

            if doc_type in ["Invoice only", "Both receipt and invoice"]:
                st.markdown("**Upload Invoice(s)**")
                if doc_count == "Single document":
                    i_file = st.file_uploader(
                        "Upload invoice",
                        type=["jpg", "jpeg", "png", "pdf", "bmp"],
                        key="invoice_single"
                    )
                    if i_file:
                        invoice_files = [i_file]
                else:
                    i_files = st.file_uploader(
                        "Upload invoices",
                        type=["jpg", "jpeg", "png", "pdf", "bmp"],
                        accept_multiple_files=True,
                        key="invoice_multi"
                    )
                    if i_files:
                        invoice_files = i_files

        else:
            if doc_type in ["Receipt only", "Both receipt and invoice"]:
                st.markdown("**Receipt URL(s)**")
                if doc_count == "Single document":
                    r_url = st.text_input("Receipt URL", placeholder="https://example.com/receipt.jpg")
                    if r_url:
                        receipt_urls = [r_url]
                else:
                    r_text = st.text_area("Receipt URLs — one per line", height=100, key="receipt_urls_text")
                    if r_text:
                        receipt_urls = [u.strip() for u in r_text.strip().split("\n") if u.strip()]

            if doc_type in ["Invoice only", "Both receipt and invoice"]:
                st.markdown("**Invoice URL(s)**")
                if doc_count == "Single document":
                    i_url = st.text_input("Invoice URL", placeholder="https://example.com/invoice.pdf")
                    if i_url:
                        invoice_urls = [i_url]
                else:
                    i_text = st.text_area("Invoice URLs — one per line", height=100, key="invoice_urls_text")
                    if i_text:
                        invoice_urls = [u.strip() for u in i_text.strip().split("\n") if u.strip()]

            with st.expander("Need sample URLs to test?"):
                st.markdown("**Sample receipt URL:**")
                st.code("https://raw.githubusercontent.com/Azure-Samples/cognitive-services-REST-api-samples/master/curl/form-recognizer/rest-api/receipt.png")
                st.markdown("**Sample invoice URL:**")
                st.code("https://raw.githubusercontent.com/Azure-Samples/cognitive-services-REST-api-samples/master/curl/form-recognizer/rest-api/invoice.pdf")

        # Step 4
        st.markdown("#### Step 4 — Read documents")

        # ── Helper: get price from item field ─────────────────
        def get_price(f2):
            price_field = (
                f2.get("TotalPrice") or
                f2.get("Price") or
                f2.get("Amount") or
                {}
            )
            if not isinstance(price_field, dict):
                return "", "$"
            price = price_field.get("valueCurrency", {})
            if not isinstance(price, dict):
                return "", "$"
            return price.get("amount", ""), price.get("currencySymbol", "$")

        # ── Helper: get total from fields ─────────────────────
        def get_total(fields):
            total_field = fields.get("Total") or fields.get("Subtotal") or {}
            if not isinstance(total_field, dict):
                return "N/A", "$"
            total = total_field.get("valueCurrency", {})
            if not isinstance(total, dict):
                return "N/A", "$"
            return total.get("amount", "N/A"), total.get("currencySymbol", "$")

        # ── Helper: display receipt ───────────────────────────
        def display_receipt(result, idx, label=""):
            for doc in result.documents:
                fields   = doc.fields
                merchant = fields.get("MerchantName", {}).get("valueString", "N/A") if fields.get("MerchantName") else "N/A"
                date     = fields.get("TransactionDate", {}).get("valueDate", "N/A") if fields.get("TransactionDate") else "N/A"
                amt, sym = get_total(fields)

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Merchant", merchant)
                with col2:
                    st.metric("Date", str(date))
                with col3:
                    st.metric("Total", f"{sym}{amt}")

                receipt_data = {
                    "merchant": merchant,
                    "date":     str(date),
                    "total":    f"{sym}{amt}",
                    "items":    []
                }

                if "Items" in fields:
                    st.markdown("**Line items:**")
                    for item in fields["Items"].get("valueArray", []):
                        f2      = item.get("valueObject", {})
                        name    = f2.get("Description", {}).get("valueString", "Unknown") if f2.get("Description") else "Unknown"
                        a2, s2  = get_price(f2)
                        price_display = f"{s2}{a2}" if a2 else "—"
                        st.markdown(f"- **{name}** — {price_display}")
                        receipt_data["items"].append({"item": name, "price": price_display})

                st.download_button(
                    f"Download receipt {idx+1} JSON",
                    data=json.dumps(receipt_data, indent=2),
                    file_name=f"receipt_{idx+1}.json",
                    mime="application/json",
                    key=f"dl_receipt_{idx}_{label}"
                )

        # ── Helper: display invoice ───────────────────────────
        def display_invoice(result, idx, label=""):
            for doc in result.documents:
                fields   = doc.fields
                vendor   = fields.get("VendorName", {}).get("valueString", "N/A") if fields.get("VendorName") else "N/A"
                inv_id   = fields.get("InvoiceId", {}).get("valueString", "N/A") if fields.get("InvoiceId") else "N/A"
                inv_date = fields.get("InvoiceDate", {}).get("valueDate", "N/A") if fields.get("InvoiceDate") else "N/A"
                due      = fields.get("DueDate", {}).get("valueDate", "N/A") if fields.get("DueDate") else "N/A"
                total_f  = fields.get("InvoiceTotal") or {}
                total    = total_f.get("valueCurrency", {}) if isinstance(total_f, dict) else {}
                amt      = total.get("amount", "N/A") if isinstance(total, dict) else "N/A"
                sym      = total.get("currencySymbol", "$") if isinstance(total, dict) else "$"

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Vendor", vendor)
                    st.metric("Invoice ID", inv_id)
                with col2:
                    st.metric("Date", str(inv_date))
                    st.metric("Due Date", str(due))
                st.metric("Total Amount", f"{sym}{amt}")

                invoice_data = {
                    "vendor": vendor, "invoice_id": inv_id,
                    "date": str(inv_date), "due_date": str(due),
                    "total": f"{sym}{amt}", "items": []
                }

                if "Items" in fields:
                    st.markdown("**Line items:**")
                    for item in fields["Items"].get("valueArray", []):
                        f2   = item.get("valueObject", {})
                        desc = f2.get("Description", {}).get("valueString", "Unknown") if f2.get("Description") else "Unknown"
                        af   = f2.get("Amount") or {}
                        af_c = af.get("valueCurrency", {}) if isinstance(af, dict) else {}
                        a2   = af_c.get("amount", "") if isinstance(af_c, dict) else ""
                        s2   = af_c.get("currencySymbol", "$") if isinstance(af_c, dict) else "$"
                        price_display = f"{s2}{a2}" if a2 else "—"
                        st.markdown(f"- **{desc}** — {price_display}")
                        invoice_data["items"].append({"description": desc, "amount": price_display})

                st.download_button(
                    f"Download invoice {idx+1} JSON",
                    data=json.dumps(invoice_data, indent=2),
                    file_name=f"invoice_{idx+1}.json",
                    mime="application/json",
                    key=f"dl_invoice_{idx}_{label}"
                )

        if st.button("Read documents", key="btn_docs", type="primary"):

            # Receipt files
            if receipt_files:
                st.markdown("---")
                st.markdown("### Receipts")
                for idx, file in enumerate(receipt_files):
                    st.markdown(f"#### Receipt {idx+1} — {file.name}")
                    with st.spinner(f"Reading receipt {idx+1}..."):
                        try:
                            file_bytes = file.read()
                            poller = di_client.begin_analyze_document(
                                model_id="prebuilt-receipt",
                                body=file_bytes,
                                content_type="application/octet-stream"
                            )
                            display_receipt(poller.result(), idx, "file")
                        except HttpResponseError as e:
                            st.error(f"Error: {e.message}")

            # Receipt URLs
            if receipt_urls:
                st.markdown("---")
                st.markdown("### Receipts")
                for idx, url in enumerate(receipt_urls):
                    st.markdown(f"#### Receipt {idx+1}")
                    with st.spinner(f"Reading receipt {idx+1}..."):
                        try:
                            poller = di_client.begin_analyze_document(
                                model_id="prebuilt-receipt",
                                body={"urlSource": url}
                            )
                            display_receipt(poller.result(), idx, "url")
                        except HttpResponseError as e:
                            st.error(f"Error: {e.message}")

            # Invoice files
            if invoice_files:
                st.markdown("---")
                st.markdown("### Invoices")
                for idx, file in enumerate(invoice_files):
                    st.markdown(f"#### Invoice {idx+1} — {file.name}")
                    with st.spinner(f"Reading invoice {idx+1}..."):
                        try:
                            file_bytes = file.read()
                            poller = di_client.begin_analyze_document(
                                model_id="prebuilt-invoice",
                                body=file_bytes,
                                content_type="application/octet-stream"
                            )
                            display_invoice(poller.result(), idx, "file")
                        except HttpResponseError as e:
                            st.error(f"Error: {e.message}")

            # Invoice URLs
            if invoice_urls:
                st.markdown("---")
                st.markdown("### Invoices")
                for idx, url in enumerate(invoice_urls):
                    st.markdown(f"#### Invoice {idx+1}")
                    with st.spinner(f"Reading invoice {idx+1}..."):
                        try:
                            poller = di_client.begin_analyze_document(
                                model_id="prebuilt-invoice",
                                body={"urlSource": url}
                            )
                            display_invoice(poller.result(), idx, "url")
                        except HttpResponseError as e:
                            st.error(f"Error: {e.message}")

            if not any([receipt_files, receipt_urls, invoice_files, invoice_urls]):
                st.warning("Please upload a document or enter a URL first.")

# ══════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════
st.divider()
st.caption(
    "Azure Computer Vision Toolkit · "
    "Caption · Tags · OCR · Object Detection · "
    "Dense Captions · People Detection · "
    "Receipt Reader · Invoice Reader"
)