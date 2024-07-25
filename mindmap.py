import pandas as pd
import streamlit as st
from pdfminer.high_level import extract_text
from unstructured.partition.pdf import partition_pdf  # Assuming partition_pdf is imported from a custom module
from pdfminer.high_level import extract_text

def parse_images(input_file):
    elements = partition_pdf(input_file,
                         strategy='hi_res',  # High resolution for image extraction
                         extract_images_in_pdf=True,
                         extract_image_block_types=["Image", "Table"],
                         extract_image_block_to_payload=False,
                         extract_image_block_output_dir="images"
                        )

    df = pd.DataFrame()
    for ind, elmt in enumerate(elements):
        e1 = elmt.to_dict()
        meta = e1['metadata']
        del e1['metadata']
        tdf = pd.DataFrame([e1 | meta])
        df = pd.concat([df, tdf])
        
    text_df = df.copy()

    images_df = df.loc[df['type'] == 'Image', :]
    df.reset_index(inplace=True, drop=True)
    df['prev_text'] = df['text'].shift(1)
    df['next_text'] = df['text'].shift(-1)

    df['New_Image_path'] = df.apply(lambda row: f"![]({row['image_path']})" if row['type'] == 'Image' else row['text'], axis=1)
    image_path_df = df.copy()
    
    return image_path_df, text_df



def parse_text(input_file):

    image_path_df, text_df = parse_images(input_file)

    text_df = text_df[~text_df['type'].isin(['Image', 'Table'])]

    def extract_text_from_pdf(pdf_path):
        text = extract_text(pdf_path)
        return text
    
    extracted_text = extract_text_from_pdf(input_file)
    
    lines = extracted_text.split('\n')
    extracted_df = pd.DataFrame({'text': lines})

    type_title_df = text_df[text_df.type == 'Title']
    footer_df = text_df[text_df.type.isin(['Footer', 'Header'])]

    for ind, row in footer_df.iterrows():
        extracted_df = extracted_df[extracted_df['text'] != row['text']]
    
    extracted_df['type'] = None

    for ind, row in type_title_df.iterrows():
        extracted_df.loc[extracted_df['text'] == row['text'], ['type']] = 'Title'
    
    # extracted_df[extracted_df.type == 'Title']

    extracted_df['new_text'] = extracted_df.apply(lambda row: '# ' + row['text'] if row['type'] == 'Title' else row['text'], axis=1)

    image_path_df = image_path_df[image_path_df['type'] != 'Header']

    index_list = []
    for ind, row in image_path_df.loc[image_path_df['type'] == 'Image'].iterrows():
        x = extracted_df.loc[extracted_df['text'] == row['next_text']]
        index_list.append(x.index.to_list())

    image_list = image_path_df[image_path_df.type == 'Image']['New_Image_path']
    zipped_list = zip(index_list, image_list)
    filtered_list_zip = [(index, image) for index, image in zipped_list if index]
    
    return filtered_list_zip, extracted_df


def merge_df(filtered_list_zip, extracted_df, input_file):
    merged_rows = []
    
    for x, y in filtered_list_zip:
        new_row = {'text': " ", 'type': ' ', 'new_text': y}
        merged_rows.extend([extracted_df.iloc[:x[0]+1], pd.DataFrame([new_row]), extracted_df.iloc[x[0]+1:]])

    merged_df = pd.concat(merged_rows).reset_index(drop=True)    
    out = merged_df.copy()
    out = out['new_text']
    sample_output = "\n".join(out.values)
    # print("Output file path:::",input_file.replace('.pdf', '.md'))
    with open(input_file.replace('.pdf', '.md'), 'w') as f:
        f.write(sample_output)


# if __name__ == "__main__":
#     input_file = r'E:\Mind Maps\scrape_pdf\Text Analytics for beginers.pdf'

#     filtered_list_zip, extracted_df = parse_text(input_file)
#     merge_df(filtered_list_zip, extracted_df, input_file)

def main():
    st.title('PDF Text and Image Extractor')

    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

    if uploaded_file is not None:
        with open("uploaded_file.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.write("File uploaded successfully!")

        filtered_list_zip, extracted_df = parse_text("uploaded_file.pdf")
        merge_df(filtered_list_zip, extracted_df, "uploaded_file.pdf")

        st.markdown("### Extracted Content")
        st.markdown("\n".join(extracted_df['new_text']))

        with open("uploaded_file.md", "w") as f:
            f.write("\n".join(extracted_df['new_text']))

        st.download_button(
            label="Download Markdown",
            data="\n".join(extracted_df['new_text']),
            file_name="extracted_content.md",
            mime="text/markdown"
        )

if __name__ == "__main__":
    main()
