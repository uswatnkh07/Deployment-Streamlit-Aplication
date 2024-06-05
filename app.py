import numpy as np
import streamlit as st
import joblib
from streamlit_option_menu import option_menu
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import altair as alt

st.set_page_config(
    page_title="Prediksi Kategori Rumah",
    page_icon="🏠",
)

st.set_option('deprecation.showPyplotGlobalUse', False)

st.write("""
Capstone Digital Product By Uswatun Khasanah - 2024
""")

######################################### Sidebar untuk navigasi #########################################
with st.sidebar:
    st.image('homesidebar.jpg')
    st.markdown("<h1 style='text-align: center;'>Housing Kategory Prediction</h1>", unsafe_allow_html=True)

with st.sidebar:
    selected = option_menu('Menu',
                           ['Home',
                            'Data Visualization',
                            'Predict'],

                            icons = ['house-fill', 
                                     'database-fill',
                                     'stars'],
                            default_index = 0)


######################################### Home #########################################
if selected == 'Home':
    st.markdown("<h1 style='text-align: center;'><b>Klasifikasi Kategori Rumah Berdasarkan Fitur-Fitur Properti untuk Evaluasi Fasilitas Furnishing</b></h1>", unsafe_allow_html=True)

    # Menampilkan gambar dari file lokal
    from PIL import Image
    image = Image.open('home.jpg')
    st.image(image, caption='')
    st.markdown("<h2>💡 Business Problem Identification</h2>", unsafe_allow_html=True)
    st.write("""
               Situasi bisnis yang mendasari analisis ini adalah kebutuhan akan pemahaman yang lebih baik tentang preferensi dan kebutuhan pasar terkait dengan fitur-fitur properti dan fasilitas furnishing dalam industri properti. 
                Kurangnya pemahaman ini dapat berdampak pada strategi pemasaran dan pengembangan properti yang tidak sesuai dengan harapan pasar. 
                Oleh karena itu, kami melakukan analisis untuk mengavaluasi dan memberikan wawasan yang lebih baik kepada pengembang properti, sehingga mereka dapat merencanakan dan memasarkan properti mereka dengan lebih efektif.

                Dengan pemahaman yang lebih baik tentang preferensi pasar, kami dapat mengambil langkah-langkah yang tepat untuk mempertahankan kepercayaan dan kepuasan pelanggan serta meminimalisir potensi kerugian yang bisa timbul akibat ketidakcocokan antara produk dan harapan konsumen. 
                Oleh karena itu, kami mengadopsi pendekatan untuk mengklasifikasikan kategori rumah berdasarkan fitur-fitur properti dan mengevaluasi preferensi pasar terkait dengan fasilitas furnishing dalam rumah. 
            """)
    
    # Spacer
    st.write("")
    st.write("")
    
    st.markdown("<h2>💡 Insight Form Data</h2>", unsafe_allow_html=True)
    st.write("""
               Dari permasalahan yang ada tentu saja solusi akhir yang menjadi hasil akhir nantinya sangat diharapkan.
             Namun sebelum itu, perlu untuk memahami, mengevaluasi, dan meningkatkan kinerja model dan analisis data.
             Oleh karena itu, perlu untuk memahami model analisis agar mencapai akhir analisis yang baik dan akurat.
              """)
    tab0, tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["Home", "Dataset", "Confusion Metrics", "Performance Metrics", "ROC Curve", "Cross-Validation Scores", "Final Model", "Cluster Category"])
    with tab0:
        def daftar_evaluasi():
            st.markdown("<h5>Daftar Evaluasi 📈</h5>", unsafe_allow_html=True)
            st.write("""Berikut adalah daftar dari evaluasi terkait matrix dan model untuk mengukur kinerja model dan menganalisis hasilnya untuk mendapatkan insight yang relevan dengan data""")
            st.write("""
                    1. **Confusion Matrix** : Digunakan untuk menggambarkan performa model klasifikasi pada set data uji, di mana nilai sebenarnya dari kelas digabungkan dengan nilai yang diprediksi oleh model.
                """)
            st.write("""
                    2. **Performance Metrics (Metrik Kinerja)** : Menggunakan metrik evaluasi yang mengukur kinerja model.
                """)
            st.write("""
                    3. **ROC Curve (Receiver Operating Characteristic Curve)** : Menggunakan kurva yang menggambarkan perbandingan antara nilai True Positive Rate (TPR) dan False Positive Rate (FPR) pada berbagai nilai ambang batas. 
            """)
            st.write("""
                    4. **Cross-Validation Scores (Skor Validasi Silang)** : Menggunaakan teknik validasi model yang membagi data menjadi subset yang saling tumpang tindih, melatih model pada beberapa subset, dan menguji model pada subset lainnya.
            """)
            st.write("""
                    5. **Cluster Category (Kategori Klaster)** : Dalam konteks analisis klaster, kita menggunakan algoritma klastering untuk mengelompokkan data ke dalam kelompok-kelompok yang homogen yaitu kategory rumah yang terbagi atas Chep, Medium, dan Expensive.
                """)
        daftar_evaluasi()

    with tab1:
            # Load data
            df = pd.read_csv('Housing.csv')
            df1 = pd.read_csv('Cleaned_Housing.csv')
            option = st.selectbox(
                "Pilih informasi apa yang ingin di dapatkan.",
                ["Lihat Informasi Dataset Awal", "Lihat Informasi Dataset Akhir"],
                help="Diberikan informasi yang mencakup visualisasi beserta keterangan atau pernyataan yang mendukung lainnya"
            )
            # composition 1
            if option == "Lihat Informasi Dataset Awal":
                def composition_plot1():
                    st.markdown("<h6 style='text-align: center;'>Dataset Awal</h6>", unsafe_allow_html=True)
                    st.write(df)
                    col1, col2 = st.columns(2)
                    with col1.expander("Informasi Selengkapnya ⓘ"):
                        st.subheader("Keterangan")
                        st.write("""
                            Ini adalah dataset public yang terdiri dari 545 baris data dan 13 kolom yang mengandung informasi mengenai kolom "price", "area", "bedrooms", "bathrooms", "stories", dan "parking" memiliki tipe data integer (int64) dan juga kolom "mainroad", "guestroom", "basement", "hotwaterheating", "airconditioning", "prefarea", dan "furnishingstatus" memiliki tipe data object, yang mungkin merupakan tipe data string atau categorical.
                        """)
                composition_plot1()

            # composition 2
            elif option == "Lihat Informasi Dataset Akhir":
                def composition_plot1():
                    st.markdown("<h6 style='text-align: center;'>Dataset Akhir</h6>", unsafe_allow_html=True)
                    st.write(df)
                    col1, col2 = st.columns(2)
                    with col1.expander("Informasi Selengkapnya ⓘ"):
                        st.subheader("Keterangan")
                        st.write("""
                            Ini adalah dataset akhir yang telah melalui proses verify data quality yaitu pengecekan kualiatas sebuah data apakah terdapat masalah seperti missing values, outliers values, duplicated values, inconsistent values / noise. Yang kemudian dilanjut dengan proses data preparation, dengan melakukan pemebersihan terhadap masalah-masalah yang mempengaruhi kualitas data tersebut. Setelah itu barulah kemudian dilakukan rekayasa fitur, mengurangi fitur - fitur yang kurang relevan, mapping, serta encoding jika dibutuhkan. Dan yang terakhir adalah mengonversi dataframe tersebut ke dalam format Cleaned_Housing.csv
                        """)
                composition_plot1()
                
########################################### TAB2
    with tab2:
        def relationship_plot():
            st.markdown("<h6 style='text-align: center;'><b>Confusion Metrics</b></h6>", unsafe_allow_html=True)

            # Menampilkan gambar dari file lokal
            from PIL import Image
            image = Image.open('Confusion_Metrics.png')
            st.image(image, caption='')

            col1, col2 = st.columns(2)
            with col1.expander("Interpretasi ⓘ"):
                st.subheader("Interpretasi")
                st.write("""
                            Gambar di atas menyoroti perbandingan kinerja dua algoritma klasifikasi, yaitu K-Nearest Neighbors (KNN) dan Gaussian Naive Bayes (GNB), berdasarkan pada matriks yang dihasilkan. Meskipun KNN memiliki nilai diagonal yang lebih tinggi, menunjukkan bahwa ia mampu mengklasifikasikan lebih banyak data dengan benar, perlu dikritisi bahwa evaluasi kinerja suatu model tidak dapat dilakukan hanya berdasarkan pada nilai diagonal matriks saja. Penting untuk melihat secara komprehensif berbagai metrik kinerja seperti akurasi, presisi, recall, dan F1-score untuk mendapatkan pemahaman yang lebih lengkap tentang keunggulan dan kelemahan relatif dari setiap model. Selain itu, perlu juga dipertimbangkan konteks spesifik dari aplikasi dan tujuan analisis, serta karakteristik dari dataset yang digunakan, karena suatu algoritma yang lebih baik dalam satu situasi mungkin tidak selalu berlaku secara umum di semua kasus.
                        """)
            with col2.expander("Actionable insight ⓘ"):
                st.subheader("Actionable insight")
                st.write("""
                            Actionable yang dapat diambil adalah pentingnya melakukan evaluasi kinerja model secara komprehensif dengan mempertimbangkan berbagai metrik kinerja serta memperhatikan konteks spesifik dari aplikasi dan tujuan analisis, sehingga dapat membuat keputusan yang lebih tepat dalam memilih algoritma klasifikasi yang sesuai untuk situasi yang ada, maka kami melakukan evaluasi kinerja model secara menyeluruh dengan mempertimbangkan berbagai metrik kinerja seperti akurasi, presisi, recall, dan f1-score.
                        """)
        relationship_plot()

################################ TAB3
    with tab3:
            # Load data
            df = pd.read_csv('Housing.csv')
            df1 = pd.read_csv('Cleaned_Housing.csv')
            option = st.selectbox(
                "Pilih informasi apa yang ingin di dapatkan.",
                ["Metrik Kinerja Model Gaussian Naive Bayes", "Metrik Kinerja Model K-Nearest Neighbors"],
                help="Diberikan informasi yang mencakup visualisasi beserta keterangan atau pernyataan yang mendukung lainnya"
            )
            # composition 1
            if option == "Metrik Kinerja Model Gaussian Naive Bayes":
                def composition_plot1():
                    st.markdown("<h6 style='text-align: center;'>Model Gaussian Naive Bayes</h6>", unsafe_allow_html=True)
                    # Menampilkan gambar dari file lokal
                    from PIL import Image
                    image = Image.open('Gaussian_Naive_Bayes_Model.png')
                    st.image(image, caption='')
                    col1, col2 = st.columns(2)
                    with col1.expander("Interpretasi ⓘ"):
                        st.subheader("Interpretasi")
                        st.write("""
                            Hasil evaluasi kinerja Gaussian Naive Bayes (GNB) menunjukkan tingkat akurasi yang cukup tinggi, namun perlu diperhatikan bahwa akurasi saja tidak mencukupi untuk memberikan gambaran yang lengkap tentang performa model. Meskipun akurasi dapat memberikan indikasi tentang seberapa baik model dapat mengklasifikasikan data secara keseluruhan, penting untuk mempertimbangkan metrik lain seperti precision, recall, dan F1-score. Dalam kasus ini, meskipun akurasi GNB cukup tinggi, nilai precision dan recall menunjukkan bahwa model cenderung memiliki keseimbangan yang baik antara presisi dan recall untuk semua kelas. Namun, F1-score yang lebih rendah mungkin menandakan bahwa ada ketidakseimbangan antara presisi dan recall untuk beberapa kelas tertentu. Selain itu, ROC-AUC Score yang mencakup multiclass juga memberikan wawasan tambahan tentang kemampuan model dalam membedakan antara kelas-kelas yang berbeda. Oleh karena itu, sementara GNB dapat memberikan hasil yang baik secara keseluruhan, analisis yang lebih mendalam dan evaluasi lebih lanjut diperlukan untuk memahami keunggulan dan kelemahan spesifik dari model ini dalam konteks aplikasi yang relevan.
                        """)
                    with col2.expander("Actionable Insight ⓘ"):
                        st.subheader("Actionable Insight")
                        st.write("""
                            Dari hasil evaluasi kinerja Gaussian Naive Bayes (GNB), terlihat bahwa model ini memiliki potensi untuk digunakan dalam pengklasifikasian data dengan tingkat akurasi yang memadai. Namun, untuk meningkatkan kinerja model secara keseluruhan, kami melakukan penyesuaian lebih lanjut terutama terkait dengan penanganan ketidakseimbangan kelas yang mungkin terjadi. Dengan melakukan penyesuaian yang sesuai, GNB memiliki potensi untuk menjadi alat yang lebih efektif dalam mengklasifikasikan data dan dapat memberikan wawasan yang lebih baik dalam pengambilan keputusan di berbagai bidang aplikasi.
                        """)
                composition_plot1()

            # composition 2
            elif option == "Metrik Kinerja Model K-Nearest Neighbors":
                def composition_plot1():
                    st.markdown("<h6 style='text-align: center;'>Model K-Nearest Neighbors</h6>", unsafe_allow_html=True)
                    # Menampilkan gambar dari file lokal
                    from PIL import Image
                    image = Image.open('KNearest_Neighbors_Model.png')
                    st.image(image, caption='')
                    col1, col2 = st.columns(2)
                    with col1.expander("Interpretasi ⓘ"):
                        st.subheader("Interpretasi")
                        st.write("""
                            Dari hasil evaluasi model Gaussian Naive Bayes (GNB), meskipun akurasi model cukup tinggi sebesar 75.23%, nilai presisi, recall, dan F1-score yang relatif rendah menunjukkan bahwa model cenderung memiliki kinerja yang lebih baik dalam mengklasifikasikan kelas mayoritas daripada kelas minoritas. Hal ini ditandai dengan nilai presisi sebesar 54.41%, yang mengindikasikan proporsi positif yang diprediksi dengan benar relatif rendah, dan nilai recall sebesar 57.19%, yang menunjukkan bahwa model memiliki kecenderungan untuk melewatkan sejumlah besar positif aktual. ROC-AUC score yang cukup tinggi sebesar 0.78 menunjukkan bahwa model memiliki kemampuan yang baik untuk membedakan antara kelas positif dan negatif. Namun demikian, untuk meningkatkan kinerja model secara keseluruhan, perlu dilakukan penyesuaian lanjutan terutama terkait dengan penanganan ketidakseimbangan kelas, seperti dengan menerapkan teknik resampling atau menggunakan metode klasifikasi yang lebih adaptif terhadap distribusi kelas yang tidak seimbang.
                        """)
                    with col2.expander("Actionable Insight ⓘ"):
                        st.subheader("Actionable Insight")
                        st.write("""
                            Untuk meningkatkan kinerja model, langkah yang kami ambil adalah melakukan penyesuaian pada parameter model, seperti menyesuaikan nilai k untuk KNN dan mencoba metode klasifikasi lain yang lebih adaptif terhadap distribusi kelas yang tidak seimbang.
                        """)
                composition_plot1()

###################################### TAB4
    with tab4:
        def relationship_plot():
            st.markdown("<h6 style='text-align: center;'><b>ROC Curve</b></h6>", unsafe_allow_html=True)

            # Menampilkan gambar dari file lokal
            from PIL import Image
            image = Image.open('ROC_Curve.png')
            st.image(image, caption='')

            col1, col2 = st.columns(2)
            with col1.expander("Interpretasi ⓘ"):
                st.subheader("Interpretasi")
                st.write("""Gambar di atas menunjukkan dua kurva ROC (Receiver Operating Characteristic) untuk dua model klasifikasi yang berbeda, yaitu Gaussian Naive Bayes (GNB) dan K-Nearest Neighbors (KNN). Kurva ROC digunakan untuk mengevaluasi kinerja model klasifikasi dalam membedakan antara dua kelas. Berdasarkan gambar, kurva ROC untuk GNB menunjukkan kinerja yang lebih baik daripada kurva ROC untuk KNN. Hal ini dapat dilihat dari beberapa hal berikut:""")
                st.write("""
                        1. **Luas area di bawah kurva (AUC)** : AUC untuk GNB adalah 0.95, sedangkan AUC untuk KNN adalah 0.60. Semakin tinggi nilai AUC, semakin baik kinerja model klasifikasi.
                    """)
                st.write("""
                        2. **Jarak kurva ROC dari diagonal** : Kurva ROC untuk GNB berada lebih jauh dari diagonal dibandingkan dengan kurva ROC untuk KNN. Hal ini menunjukkan bahwa GNB lebih baik dalam membedakan antara dua kelas dibandingkan dengan KNN.
                    """)
                st.write("""
                        3. **Titik potong kurva ROC dengan sumbu Y** : Titik potong kurva ROC dengan sumbu Y untuk GNB adalah 1.0, sedangkan titik potong kurva ROC dengan sumbu Y untuk KNN adalah 0.90. Hal ini menunjukkan bahwa GNB memiliki tingkat true positive rate (TPR) yang lebih tinggi pada false positive rate (FPR) yang rendah dibandingkan dengan KNN. 
                """)
                st.write("""
                        Secara keseluruhan, interpretasi kritis dari gambar hasil ROC di atas adalah bahwa GNB merupakan model klasifikasi yang lebih baik daripada KNN untuk tugas yang spesifik ini. GNB memiliki AUC yang lebih tinggi, kurva ROC yang lebih jauh dari diagonal, dan tingkat TPR yang lebih tinggi pada FPR yang rendah.
                        """)
            with col2.expander("Actionable insight ⓘ"):
                st.subheader("Actionable insight")
                st.write("""
                            Berdasarkan interpretasi hasil roc di atas, tindakan yang dapat diambil adalah menganalisis data pelatihan dan memastikan data pelatihan memiliki kualitas yang baik dan cukup representatif dari populasi target. Kemudian menyesuaikan parameter model untuk meningkatkan kinerja model. Dan yang terakhir melakukan validasi silang untuk memastikan bahwa hasil evaluasi model dapat digeneralisasi ke data baru.
                        """)
        relationship_plot()

    ###################################### TAB5
    with tab5:
        def relationship_plot():
            st.markdown("<h6 style='text-align: center;'><b>Cross-Validation Scores</b></h6>", unsafe_allow_html=True)

            # Menampilkan gambar dari file lokal
            from PIL import Image
            image = Image.open('Cross _Validation.png')
            st.image(image, caption='')

            col1, col2 = st.columns(2)
            with col1.expander("Interpretasi ⓘ"):
                st.subheader("Interpretasi")
                st.write("""
                            Meskipun score gaussian naive bayes dibandingkan dengan k-nearest neighbors, tetapi setelah di analisa dan dikritisi lebih anjut terlihat bahwa gaussian naive bayes adalah model yang lebih baik untuk tugas klasifikasi yang diwakili dalam grafik. Hal ini karena gaussian naive bayes memiliki kinerja yang lebih baik secara keseluruhan dan lebih konsisten dalam kinerjanya di semua lipatan.
                        """)
            with col2.expander("Actionable insight ⓘ"):
                st.subheader("Actionable insight")
                st.write("""
                            Meskipun skor Gaussian Naive Bayes lebih tinggi daripada K-Nearest Neighbors, hasil analisis yang lebih mendalam menunjukkan bahwa Gaussian Naive Bayes adalah pilihan yang lebih baik untuk tugas klasifikasi yang diwakili dalam grafik. Ini karena model tersebut menunjukkan kinerja yang lebih baik secara keseluruhan dan lebih konsisten dalam semua lipatan data. Oleh karena itu, langkah yang dapat diambil adalah mengadopsi model Gaussian Naive Bayes untuk aplikasi klasifikasi ini. Selanjutnya, penting untuk melakukan pemantauan terus-menerus terhadap kinerja model ini dan memperbarui model sesuai dengan perkembangan data dan kebutuhan bisnis yang berkelanjutan.
                        """)
        relationship_plot()

    with tab6:
            # Load data
            df = pd.read_csv('Housing.csv')
            df1 = pd.read_csv('Cleaned_Housing.csv')
            option = st.selectbox(
                "Pilih informasi apa yang ingin di dapatkan.",
                ["Keputusan Final Model", "Perbandingan Akurasi Gaussian Naive Bayes Sebelum dan Sesudah Tuning", "Perbandingan Akurasi K-Nearest Neighbor Sebelum dan Sesudah Tuning"]
            )
            # composition 1
            if option == "Keputusan Final Model":
                def composition_plot1():
                    st.markdown("<h6 style='text-align: center;'>Gaussian Naive Bayes Sebagai Model</h6>", unsafe_allow_html=True)
                    st.write("""
                            Menjadi keputusan besar memang dalam hal menentukan model apa yang akan digunakan. Namun dari hasil evaluasi yang telah dilakukan, langkah selanjutnya adalah melakukan pemilihan model yang optimal untuk klasifikasi kategori rumah berdasarkan fitur-fitur properti. Berdasarkan analisis, model Gaussian Naive Bayes menunjukkan akurasi yang cukup baik dengan performa yang stabil dalam cross-validation. Selain itu, dilakukan tuning hyperparameter untuk memperbaiki performa model, dimana akurasi setelah tuning mengalami peningkatan yang signifikan. Langkah berikutnya adalah menerapkan model Gaussian Naive Bayes yang telah dioptimalkan untuk memprediksi kategori furnishing berdasarkan fitur-fitur yang telah dikelompokkan sebelumnya menggunakan algoritma K-Means clustering.
                        """)
                composition_plot1()

            # composition 2
            elif option == "Perbandingan Akurasi Gaussian Naive Bayes Sebelum dan Sesudah Tuning":
                def composition_plot1():
                    st.markdown("<h6 style='text-align: center;'>Akurasi Gaussian Naive Bayes Sebelum dan Sesudah Tuning</h6>", unsafe_allow_html=True)
                    # Menampilkan gambar dari file lokal
                    from PIL import Image
                    image = Image.open('tuningGNB.png')
                    st.image(image, caption='')
                    col1, col2 = st.columns(2)
                    with col1.expander("Informasi Selengkapnya ⓘ"):
                        st.subheader("Keterangan")
                        st.write("""
                            Ini adalah perbandingan akurasi gaussian naive bayes sebelum dan sesudah tuning. Terlihat bahwa nilai yang dihasilkan lebih baik.
                        """)
                composition_plot1()

            # composition 3
            elif option == "Perbandingan Akurasi K-Nearest Neighbor Sebelum dan Sesudah Tuning":
                def composition_plot1():
                    st.markdown("<h6 style='text-align: center;'>Akurasi K-Nearest Neighbor Sebelum dan Sesudah Tuning</h6>", unsafe_allow_html=True)
                    # Menampilkan gambar dari file lokal
                    from PIL import Image
                    image = Image.open('tuningKNN.png')
                    st.image(image, caption='')
                    col1, col2 = st.columns(2)
                    with col1.expander("Informasi Selengkapnya ⓘ"):
                        st.subheader("Keterangan")
                        st.write("""
                            Ini adalah perbandingan akurasi K-Nearest Neighbor sebelum dan sesudah tuning. Terlihat bahwa nilai yang dihasilkan lebih kurang baik.
                        """)
                composition_plot1()
    ###################################### TAB7
    with tab7:
            def relationship_plot():
                st.markdown("<h6 style='text-align: center;'><b>Cluster Categories</b></h6>", unsafe_allow_html=True)

                # Menampilkan gambar dari file lokal
                from PIL import Image
                image = Image.open('Cluster_Categories.png')
                st.image(image, caption='')

                col1, col2 = st.columns(2)
                with col1.expander("Interpretasi ⓘ"):
                    st.subheader("Interpretasi")
                    st.write("""
                                Gambar tersbut menyoroti sebaran datakategori rumah yang terbagi menjadi tiga yaitu, kategori Cheap sebanyak 55,2 %, Medium sebanyak 12,5%, dan yang terakhir yaitu Expensive sebanyak 32,3%. Sebaran data tersebut tentu berdasarkan hasil clustering berdasarkan rata-rata Harga property. Dan data inilah yang kemudian digunakan sebagai target dalam model predict. Dan kami rasa ini penting ada, untuk menciptakan keselarasan antara problem yang ada, dengan solusi yang diberikan yaitu mengkategorikan dan memprediksi kategori rumah atau property.
                            """)
                with col2.expander("Actionable insight ⓘ"):
                    st.subheader("Actionable insight")
                    st.write("""
                                 Langkah yang dapat diambil adalah dengan memastikan bahwa metrik evaluasi yang digunakan dapat memberikan gambaran yang seimbang tentang kinerja model terhadap setiap kategori, seperti menggunakan F1-score atau AUC-PR untuk mendapatkan pemahaman yang lebih lengkap tentang kemampuan model dalam memprediksi setiap kategori dengan adil.
                            """)
            relationship_plot()

    # Spacer
    st.write("")
    st.write("")

    st.markdown("<h3>💡 Solution Form Insight</h3>", unsafe_allow_html=True)
    st.write("""Solusi akhir dari analisis yang berjudul "Klasifikasi Kategori Rumah Berdasarkan Fitur-Fitur Properti untuk Evaluasi Fasilitas Furnishing" ada tiga, diantaranya yaitu Mengklasterisasi Kategori Rumah Berdasarkan Fitur-fitur Properti, Mengevaluasi Preferensi Pasar terkait Fasilitas Furnishing,  dan Memprediksi Kategori Rumah untuk Inputan yang Diberikan. Berikut informasi lengkapnya""")
    st.write("""
                ✔️ **Mengklasterisasi Kategori Rumah Berdasarkan Fitur-fitur Properti** : 
             Kami melakukan analisis klasterisasi menggunakan algoritma K-means untuk mengelompokkan rumah-rumah berdasarkan fitur-fitur properti seperti harga, luas area, jumlah kamar tidur, jumlah kamar mandi, jumlah lantai, dan fitur-fitur lainnya.
             Hasilnya, rumah-rumah dibagi menjadi 3 klaster utama: Cheap, Medium, dan Expensive, berdasarkan pada harga rata-rata dari setiap klaster.
                """)
    st.write("""
                ✔️ **Mengevaluasi Preferensi Pasar terkait Fasilitas Furnishing** :
             Kami menggunakan analisis klaster untuk memahami preferensi pasar terkait dengan fasilitas furnishing dalam rumah.
             Setiap klaster memiliki karakteristik unik dalam hal preferensi dan kebutuhan pasar terkait dengan fasilitas furnishing seperti akses jalan utama (mainroad), kamar tamu (guestroom), basement, pemanas air (hotwaterheating), penyejuk udara (airconditioning), dan tempat parkir (parking).
                """)
    st.write("""
                ✔️ **Memprediksi Kategori Rumah untuk Inputan yang Diberikan** :
             Dengan memanfaatkan model yang telah kita latih menggunakan K-means, kami dapat memprediksi kategori rumah (Cheap, Medium, atau Expensive) untuk rumah baru berdasarkan fitur-fitur properti yang diberikan.
             Inputan fitur-fitur properti akan dievaluasi dan diklasifikasikan ke dalam salah satu kategori rumah berdasarkan kesesuaian fitur-fitur tersebut dengan karakteristik masing-masing klaster.
            """)
    st.write("""
             Dengan pendekatan ini, pengembang properti dapat memiliki pemahaman yang lebih baik tentang preferensi dan kebutuhan pasar, sehingga mereka dapat merencanakan dan memasarkan properti mereka dengan lebih efektif. 
             Dengan mengklasifikasikan rumah-rumah ke dalam kategori yang sesuai, pengembang dapat meminimalkan ketidakcocokan antara produk dan harapan konsumen, dan pada akhirnya mempertahankan kepercayaan dan kepuasan pelanggan serta meminimalisir potensi kerugian yang bisa timbul.
             """)
    
    
######################################### VISUALISASI #########################################
if selected == 'Data Visualization':
   st.markdown("<h2>💡 Visualisasi</h2>", unsafe_allow_html=True)

   st.write("""
            Pada bagian ini, menampilkan gambar visualisasi yang informatif lengkap dengan interpretasi dan juga insight, mengenai
            faktor-faktor yang berpengaruh terhadap keputusan pembelian property tertentu oleh konsumen.
      """)

   tab1, tab2, tab3, tab4, tab5 = st.tabs(["Home", "Composition", "Relationship", "Distribution", "Comparison"])

   with tab1:
      def daftar_visualisasi():
         st.markdown("<h4>Daftar Visualisasi 📈</h4>", unsafe_allow_html=True)
         st.write("""Berikut adalah daftar dari visualisasi terkait faktor-faktor yang mempengaruhi prediksi minat mahasiswa dalam menentukan 
                  karir berwirausaha. Silahkan pilih tab untuk melihat visualisasi data!""")
         st.write("""
                  1. **Composition** : komposisi variabel kategorikal untuk memberi gambaran tentang proporsi rumah dengan berbagai fitur.
            """)
         st.write("""
                  2. **Relationship** : hubungan antara variabel numerik melihat korelasi antara variabel-variabe.
            """)
         st.write("""
                  3. **Distribution** : distribusi dari variabel numerik dalam dataset untuk memberi pemahaman tentang bagaimana nilai-nilai variabel tersebar di dalam dataset. 
         """)
         st.write("""
                  4. **Comparison** : membandingkan berbagai fitur properti dengan harga untuk memahami bagaimana harga dipengaruhi oleh berbagai fitur properti dan apakah terdapat perbedaan signifikan antara properti dengan berbagai fitur.
         """)
      daftar_visualisasi()

   with tab2:
      # Load data
      df = pd.read_csv('Housing.csv')

      option = st.selectbox(
         "Pilih informasi apa yang ingin di dapatkan.",
         ["Komposisi Kategorikal Demografi Rumah", "Komposisi Presentase Status Furnitur"],
         help="Diberikan informasi yang mencakup visualisasi beserta interpretasi dan insightnya."
      )
      # composition 1
      if option == "Komposisi Kategorikal Demografi Rumah":
         def composition_plot1():
            st.header("Composition")

            # List variabel kategorikal
            categorical_vars = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea', 'furnishingstatus']

            # Judul aplikasi
            st.markdown("<h6 style='text-align: center;'>Bar Chart Untuk Variabel Kategorikal</h6>", unsafe_allow_html=True)

            # Membuat plot bar chart untuk setiap variabel kategorikal
            plt.figure(figsize=(14, 10))
            for var in categorical_vars:
               plt.subplot(2, 4, categorical_vars.index(var) + 1)
               sns.countplot(data=df, x=var, color='#41ab5d')
               plt.title(f'Plot hitungan dari {var}')
               plt.xticks(rotation=45)
            
            # Menambahkan jarak antara setiap gambar
            plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.4, hspace=0.4)

            # Menampilkan plot menggunakan Streamlit
            st.pyplot(plt)

            col1, col2 = st.columns(2)
            with col1.expander("Interpretasi ⓘ"):
               st.subheader("Interpretasi")
               st.write("""
                  Gambar di atas menunjukkan jumlah data untuk beberapa kategori. Setiap kategori diwakili oleh batang dengan ketinggian yang proporsional dengan jumlah datanya. Contohnya, pada variabel "mainroad" memiliki distribusi yang signifikan dalam dataset, dengan sebagian besar properti berlokasi di dekat jalan utama. Hal ini menunjukkan bahwa akses ke jalan utama mungkin menjadi salah satu faktor yang dipertimbangkan penting oleh pemilik properti atau pembeli potensial dalam memilih lokasi properti. Dari contoh tersebut, analisis lebih lanjut perlu dilakukan untuk mendapatkan pemahaman yang lebih holistik tentang preferensi pasar dan faktor-faktor yang mempengaruhi harga atau permintaan properti.
               """)
            with col2.expander("Actionable insight ⓘ"):
               st.subheader("Actionable insight")
               st.write("""
                  Langkah berikutnya yang dapat diambil adalah melakukan analisis lebih lanjut untuk memahami dampak variabel ini terhadap harga atau permintaan properti secara keseluruhan. Tindakan lanjutan melibatkan penelusuran faktor-faktor lain yang mungkin berperan dalam menentukan nilai properti, seperti keberadaan fasilitas umum, kondisi lingkungan sekitar, dan aspek-aspek lain yang relevan bagi calon pembeli.
               """)
         composition_plot1()
         
      # composition 2
      elif option == "Komposisi Presentase Status Furnitur":
         def composition_plot2():
            st.header("Composition")

            # Judul aplikasi
            st.markdown("<h6 style='text-align: center;'>Pie Chart Presentase Status Furnitur</h6>", unsafe_allow_html=True)

            # Custom palette
            custom_palette = ['#005a32', '#41ab5d', '#a1d99b']

            # Membuat plot pie chart untuk persentase status furnitur
            fig = px.pie(df['furnishingstatus'].value_counts().reset_index(),
                        names=df['furnishingstatus'].value_counts().index,
                        values=df['furnishingstatus'].value_counts().values,
                        color_discrete_sequence=custom_palette)

            # Menampilkan plot menggunakan Streamlit
            st.plotly_chart(fig)

            col1, col2 = st.columns(2)
            with col1.expander("Interpretasi ⓘ"):
               st.subheader("Interpretasi")
               st.write("""
                  Pie Chart di atas menunjukkan persentase dari status furnitur. Kategori "semi-furnished" memiliki persentase terbesar dalam data, mencapai 41.7%. Sementara itu, kategori "unfurnished" dan "furnished" memiliki persentase masing-masing 32.7% dan 25.7%.
               """)
            with col2.expander("Actionable insight ⓘ"):
               st.subheader("Actionable insight")
               st.write("""
                  Langkah yang dapat diambil adalah mempertimbangkan untuk memprioritaskan produksi atau pemasaran lebih banyak properti dengan kondisi "semi-furnished". Hal ini dapat menjadi strategi yang efektif untuk menanggapi preferensi pasar yang dominan terhadap properti dengan furnitur yang sudah dilengkapi sebagian. Dengan fokus pada produksi lebih banyak properti "semi-furnished", pengembang properti dapat meningkatkan daya tarik properti mereka bagi mayoritas pembeli potensial, sehingga memperkuat posisi mereka di pasar dan meningkatkan kesempatan penjualan.
               """)
         composition_plot2()

   with tab3:
      st.header("Relationship")
      def relationship_plot():
         numerical_columns = df.select_dtypes(include=[np.number]).columns

         if len(numerical_columns) < 2:
            st.error("Error: At least two numerical columns are required for heatmap generation.")
         else:
            # Variabel numerik
            # Matriks korelasi
            correlation_matrix = df[numerical_columns].corr()

            # Membuat heatmap untuk melihat korelasi antara variabel numerik
            plt.figure(figsize=(8, 6))
            sns.heatmap(correlation_matrix, annot=True, cmap='BuGn', fmt=".2f")
            plt.title('Heatmap Korelasi Variabel Numerik')

            # Menampilkan heatmap menggunakan Streamlit
            st.pyplot(plt)

         col1, col2 = st.columns(2)
         with col1.expander("Interpretasi ⓘ"):
            st.subheader("Interpretasi")
            st.write("""
                        Heatmap di atas menunjukkan korelasi antara variabel-variabel numerik dalam data. Contohnya, "price" dan "area" memiliki korelasi positif menunjukkan bahwa semakin besar luas rumah semakin tinggi pula harganya.
                     """)
         with col2.expander("Actionable insight ⓘ"):
            st.subheader("Actionable insight")
            st.write("""
                        Langkah yang dapat diambil adalah mempertimbangkan hubungan yang memiliki nilai korelasi di atas 0,5 untuk dijadikan pertimbangan penting dalam pengembangan dan pemasaran properti. Korelasi yang kuat antara variabel-variabel seperti "price" dan "area" menunjukkan bahwa variabel-variabel ini saling berhubungan erat. Oleh karena itu, dalam merencanakan proyek pengembangan properti, penting untuk memperhatikan faktor-faktor yang berkorelasi tinggi, seperti luas properti, jumlah kamar, dan fasilitas tambahan, karena hal ini dapat mempengaruhi harga dan permintaan properti. Memahami korelasi ini dapat membantu pengambil keputusan untuk menyesuaikan strategi pemasaran, penetapan harga, dan pengembangan properti secara keseluruhan.
                     """)
      relationship_plot()

   with tab4:
    st.header("Distribution")
    
    def distribution_plot():
        numerical_columns = df.select_dtypes(include=[np.number]).columns

        if len(numerical_columns) < 2:
            st.error("Error: At least two numerical columns are required for heatmap generation.")
        else:
            # Judul aplikasi
            st.markdown("<h6 style='text-align: center;'>Sebaran Nilai-Nilai Variabel Numerik</h6>", unsafe_allow_html=True)

            # Distribution Analysis
            plt.figure(figsize=(16, 10))

            for i, var in enumerate(numerical_columns):
                plt.subplot(2, 3, i+1)
                sns.histplot(df[var], kde=True, color='#41ab5d')
                plt.title(f'Distribusi {var}')
                plt.xlabel(var)
                plt.ylabel('Frekuensi')

            plt.tight_layout()

            # Menampilkan plot menggunakan Streamlit
            st.pyplot(plt)

            col1, col2 = st.columns(2)
            with col1.expander("Interpretasi ⓘ"):
                st.subheader("Interpretasi")
                st.write("""
                         Meskipun histogram menampilkan distribusi frekuensi untuk variabel numerik seperti harga, luas area, jumlah kamar tidur, dan fitur lainnya, itu hanya memberikan gambaran umum tentang bagaimana data tersebar. Histogram tidak memberikan informasi detail tentang jumlah data yang mewakili setiap nilai secara spesifik. Misalnya, untuk variabel luas area, histogram dapat menunjukkan bahwa sebagian besar data memiliki luas area sekitar 4000, tetapi tidak memberikan informasi tentang jumlah data yang memiliki luas area tepat 4000. Oleh karena itu, penting untuk memahami bahwa histogram hanya memberikan gambaran umum tentang distribusi data numerik, dan tidak memberikan detail spesifik tentang jumlah data pada setiap nilai.""")
            with col2.expander("Actionable insight ⓘ"):
                st.subheader("Actionable insight")
                st.write("""
                         Langkah-langkah yang dapat diambil termasuk menggunakan metode statistik deskriptif untuk menghitung rata-rata, median, kuartil, dan rentang data untuk setiap variabel numerik. Selain itu, visualisasi tambahan seperti box plot atau density plot dapat memberikan wawasan tambahan tentang distribusi data dan adanya pencilan (outlier). Analisis lebih lanjut ini akan membantu dalam pemahaman yang lebih mendalam tentang karakteristik data numerik dan memungkinkan pengambilan keputusan yang lebih tepat dalam konteks analisis data yang sedang dilakukan.
                        """)

    distribution_plot()

   with tab5:
      st.header("Comparison")
      def comparison_plot():
         df_grouped = df.groupby(['furnishingstatus', 'mainroad']).size().unstack()

         # Plotting stacked bar plot
         fig, ax = plt.subplots(figsize=(10, 6))
         df_grouped.plot(kind='bar', stacked=True, ax=ax, color=['#a1d99b', '#41ab5d'])
         ax.set_xlabel('Status Furnitur')
         ax.set_ylabel('Jumlah')
         ax.set_title('Status Furnitur Berdasarkan Jalan Utama')
         ax.legend(title='mainroad')
         plt.xticks(rotation=45)  # Rotate x-axis labels for better visibility

         # Menampilkan plot menggunakan Streamlit
         st.pyplot(fig)

         col1, col2 = st.columns(2)
         with col1.expander("Interpretasi ⓘ"):
            st.subheader("Interpretasi")
            st.write("""
               Stacked bar plot di atas memvisualisasikan distribusi jumlah furnitur untuk properti berdasarkan akses ke jalan utama. Dari grafik tersebut, terlihat bahwa sebagian besar properti dengan status "semi-furnished" terletak di dekat jalan utama. Hal ini menunjukkan bahwa akses ke jalan utama bisa jadi menjadi faktor penting dalam menentukan tingkat furniturasi properti. Namun, perlu dicatat bahwa informasi ini hanya memberikan gambaran umum, dan analisis lebih lanjut diperlukan untuk memahami hubungan yang lebih mendalam antara status furnitur dan lokasi properti serta bagaimana faktor-faktor ini dapat memengaruhi harga atau permintaan properti.
                     """)
         with col2.expander("Actionable insight ⓘ"):
            st.subheader("Actionable insight")
            st.write("""
               
Menyelidiki apakah terdapat faktor-faktor lain yang memengaruhi status furnitur properti selain lokasi dekat dengan jalan.""")
      comparison_plot()

######################################### predict #########################################
if selected == 'Predict':
    model = joblib.load('gnb.sav')

    st.sidebar.success("Pilih page di atas.")

    st.title("📈 Analisis Prediksi")

    st.write("""
        Ini ada bagian untuk memprediksi kategori rumah (Cheap, Medium, atau Expensive) didasarkan pada fitur-fitur properti yang diberikan. Silahkan untuk memasukkan beberapa fitur properti, termasuk apakah properti dekat dengan jalan utama, memiliki kamar tamu, basement, pemanas air panas, AC, area pilihan, jumlah kamar tidur, kamar mandi, jumlah lantai, tempat parkir, harga, dan luas area. 
    """)

    st.subheader("💡 Pilih Opsi ")

    inputs = {}
    col1, col2, col3 = st.columns(3)

    with col1:
        inputs['mainroad'] = st.selectbox(
            'Apakah Dekat Dengan Jalanan?',
            options=[0, 1],
            format_func=lambda x: ["No", "Yes"][x],
        )
        inputs['guestroom'] = st.selectbox(
            'Apakah Memiliki Kamar Tamu?',
            options=[0, 1],
            format_func=lambda x: ["No", "Yes"][x],
        )
        inputs['basement'] = st.selectbox(
            'Apakah MemilikiBasement?',
            options=[0, 1],
            format_func=lambda x: ["No", "Yes"][x],
        )
        inputs['hotwaterheating'] = st.selectbox(
            'Apakah Memiliki Penghangat Air Panas?',
            options=[0, 1],
            format_func=lambda x: ["No", "Yes"][x],
        )
        inputs['airconditioning'] = st.selectbox(
            'Apakah Memiliki AC?',
            options=[0, 1],
            format_func=lambda x: ["No", "Yes"][x],
        )
        inputs['prefarea'] = st.selectbox(
            'Apakah Memiliki Area Pilihan?',
            options=[0, 1],
            format_func=lambda x: ["No", "Yes"][x],
        )
        inputs['furnishingstatus'] = st.selectbox(
            'Apa Status Furniturnya?',
            options=[0, 1, 2],
            format_func=lambda x: ["Unfurnished", "Semi-furnished", "Furnished"][x],
        )

    with col2:
        inputs['bedrooms'] = st.slider('Bedrooms', 0, 6, 2, help="Jumlah Kamar?")
        inputs['bathrooms'] = st.slider('Bathrooms', 0, 4, 1, help="Jumlah Kamar  Mandi?")
        inputs['stories'] = st.slider('Stories', 0, 4, 0, help="Jumlah Lantai?")
        inputs['parking'] = st.slider('Parking spaces', 0, 3, 0, help="Jumlah Tempat Parkir?")

    with col3:
        inputs['price'] = st.number_input('Price (Millions)', min_value=100000.00, max_value=1920500.0, step=0.01, format="%.2f", help="Harga Rumah?")
        inputs['area'] = st.number_input('Area (m²)', min_value=1650.0, max_value=15600.0, step=0.01, format="%.2f", help="Luas Area Atau Luas Tanah?")

    input_data = pd.DataFrame(inputs, index=[0], columns=model.feature_names_in_)
    if st.button('Predict'):
        # Validasi input
        if input_data is None:
            st.error("Silakan isi nilai input terlebih dahulu!")
        else:
            prediction = model.predict(input_data)
            
            if prediction == 0:
                st.success("Hasil Prediksi Masuk Ke Dalam Kategori Rumah: Cheap")
            elif prediction == 1:
                st.info("Hasil Prediksi Masuk Ke Dalam Kategori Rumah: Medium")
            elif prediction == 2:
                st.warning("Hasil Prediksi Masuk Ke Dalam Kategori Rumah: Expensive")
            else:
                raise ValueError(f"Unexpected prediction result: {prediction}")
