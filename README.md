# LAB 5 - Deep Learning trong Khoa học Dữ liệu (DS201)

## 📚 Giới thiệu

Lab 5 tập trung vào việc xây dựng và huấn luyện mô hình **Transformer Encoder** theo kiến trúc "Attention is All You Need" cho hai bài toán xử lý ngôn ngữ tự nhiên tiếng Việt:
1. **Phân loại Domain** (Domain Classification)
2. **Gán nhãn Chuỗi** (Named Entity Recognition)

---

## 📋 Nội dung Lab

### **Bài 1: Phân loại Domain trên bộ dữ liệu UIT-ViOCD**

#### 🎯 Mô tả bài toán
Xây dựng mô hình phân loại domain (lĩnh vực) của các câu bình luận tiếng Việt. Mô hình cần xác định câu bình luận thuộc domain nào trong các domain: `mobile`, `app`, `fashion`, v.v.

#### 📊 Dữ liệu
- **Nguồn**: [UIT-ViOCD](https://drive.google.com/drive/folders/1Lu9axyLkw7dMx80uLRgvCnZsmNzhJWAa?usp=sharing)
- **Cấu trúc**:
  - `train.json`: Dữ liệu huấn luyện
  - `dev.json`: Dữ liệu validation
  - `test.json`: Dữ liệu test
- **Format**:
```json
{
    "0": {
        "review": "gói hàng cẩn thận . chơi pubg...",
        "label": "non-complaint",
        "domain": "mobile"
    }
}
```

#### 🏗️ Kiến trúc mô hình
```
Input Text
    ↓
Embedding Layer (vocab_size → d_model=256)
    ↓
Positional Encoding
    ↓
Encoder Layer 1 (Multi-Head Attention + FFN)
    ↓
Encoder Layer 2 (Multi-Head Attention + FFN)
    ↓
Encoder Layer 3 (Multi-Head Attention + FFN)
    ↓
Global Average Pooling
    ↓
Classification Head (d_model → num_classes)
    ↓
Output: Domain Label
```

#### ⚙️ Cấu hình
- **d_model**: 256
- **num_heads**: 8
- **d_ff**: 1024
- **num_layers**: 3
- **max_len**: 128
- **batch_size**: 32
- **learning_rate**: 0.0001
- **dropout**: 0.1

#### 📈 Đánh giá
- **Metrics**: Accuracy, Precision, Recall, F1-score
- **Công cụ**: scikit-learn classification_report

#### 🚀 Cách chạy
```bash
# Cài đặt thư viện
pip install torch numpy scikit-learn matplotlib

# Chạy training
python transformer_classifier.py
```

#### 📁 Output
- `best_model.pt`: Model tốt nhất
- `training_curves.png`: Đồ thị loss và accuracy
- Classification report trên test set

---

### **Bài 2: Gán nhãn Chuỗi trên bộ dữ liệu PhoNER_COVID19**

#### 🎯 Mô tả bài toán
Xây dựng mô hình Named Entity Recognition (NER) để nhận diện các thực thể trong văn bản tiếng Việt liên quan đến COVID-19, bao gồm: tên người, địa điểm, ngày tháng, tổ chức, triệu chứng bệnh, v.v.

#### 📊 Dữ liệu
- **Nguồn**: [PhoNER_COVID19](https://github.com/VinAIResearch/PhoNER_COVID19)
- **Cấu trúc**:
  - `train_syllable.json`: Dữ liệu huấn luyện
  - `dev_syllable.json`: Dữ liệu validation
  - `test_syllable.json`: Dữ liệu test
- **Format**: JSON Lines (mỗi dòng 1 sample)
```json
{"words": ["Bộ", "Y", "tế", "."], "tags": ["B-ORGANIZATION", "I-ORGANIZATION", "I-ORGANIZATION", "O"]}
```

#### 🏷️ Entity Tags
| Tag | Mô tả | Ví dụ |
|-----|-------|-------|
| `B-PATIENT_ID`, `I-PATIENT_ID` | Mã bệnh nhân | 523, BN91 |
| `B-NAME`, `I-NAME` | Tên người | Nguyễn Văn A |
| `B-AGE` | Tuổi | 67 tuổi |
| `B-GENDER` | Giới tính | nam, nữ |
| `B-JOB`, `I-JOB` | Nghề nghiệp | phi công |
| `B-LOCATION`, `I-LOCATION` | Địa điểm | TP. HCM, Hà Nội |
| `B-ORGANIZATION`, `I-ORGANIZATION` | Tổ chức | Bộ Y tế |
| `B-DATE` | Ngày tháng | 31/7, ngày 14-4 |
| `B-SYMPTOM_AND_DISEASE`, `I-SYMPTOM_AND_DISEASE` | Triệu chứng/Bệnh | sốt cao, khó thở |
| `B-TRANSPORTATION`, `I-TRANSPORTATION` | Phương tiện | máy bay, taxi |
| `O` | Không phải entity | các từ khác |

#### 🏗️ Kiến trúc mô hình
```
Input Sequence
    ↓
Embedding Layer (vocab_size → d_model=256)
    ↓
Positional Encoding
    ↓
Encoder Layer 1 (Multi-Head Attention + FFN)
    ↓
Encoder Layer 2 (Multi-Head Attention + FFN)
    ↓
Encoder Layer 3 (Multi-Head Attention + FFN)
    ↓
Classification Head (d_model → num_tags) [Token-level]
    ↓
Output: Tag Sequence
```

#### ⚙️ Cấu hình

- **d_model**: 256
- **num_heads**: 8
- **d_ff**: 1024
- **num_layers**: 3
- **max_len**: 150
- **batch_size**: 16
- **learning_rate**: 0.0003
- **dropout**: 0.3
- **loss**: Focal Loss (gamma=2.0)
- **optimizer**: AdamW với weight decay
- **scheduler**: CosineAnnealingWarmRestarts

#### 📈 Đánh giá
- **Metrics**: Entity-level F1-score, Precision, Recall
- **Công cụ**: seqeval library
- **Lưu ý**: Metrics được tính theo entity hoàn chỉnh (B-I matching), không phải token-level

#### 🚀 Cách chạy

```bash
# Cài đặt thư viện
pip install torch numpy scikit-learn matplotlib seqeval

# Chạy training
python ner_improved_version.py
```

#### 📁 Output
- `best_ner_model_improved.pt`: Model tốt nhất
- `ner_training_curves_improved.png`: Đồ thị training
- Detailed classification report theo từng entity type

---

## 📂 Cấu trúc thư mục

```
LAB5_DS201/
├── README.md
│
├── bai1_domain_classification/
│   ├── transformer_classifier.py
│   ├── train.json
│   ├── dev.json
│   ├── test.json
│   ├── best_model.pt (sau khi train)
│   └── training_curves.png (sau khi train)
│
└── bai2_ner/
    ├── ner.py 
    ├── train_syllable.json
    ├── dev_syllable.json
    ├── test_syllable.json
    ├── [best_ner_model_improved.pt](https://drive.google.com/file/d/1gtxJlFeIZxAmXfff5fgKImB9OX1KXIdN/view?usp=sharing) (sau khi train)
    └── ner_training_curves_improved.png (sau khi train)
```

---

*Lab được thiết kế bởi: Lê Diễm Quỳnh Như*  
*Môn: DS201 - Deep Learning trong Khoa học Dữ liệu*  
*Học kỳ: [HK1]*  
*Năm học: [Năm 3]*
