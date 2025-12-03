Part 1: Conceptual Design

**Abstract**: Medical imaging plays a critical role in diagnosing respiratory conditions, yet accurate interpretation of chest CT scans and X-rays remains challenging. Visual differences between diseases like COVID-19 and tuberculosis can be subtle, and diagnostic interpretation often varies between radiologists. This project aims to develop an automated multi-class classification system capable of distinguishing between COVID-19, tuberculosis, and normal lung cases from chest imaging data.

**Problem**: The primary challenge in chest imaging classification lies in the subtle visual distinctions between different pathologies. COVID-19 typically presents with bilateral or peripheral ground-glass opacities, consolidation in lower lobes, vascular enlargement, and crazy-paving patterns. Tuberculosis, on the other hand, manifests through upper lobe infiltrates, cavitary lesions, tree-in-bud appearance, fibrosis, and volume loss with often asymmetrical chronic scarring. Normal lungs appear fully aerated with no visible opacities, nodules, fibrosis, cavitation, or signs of infection. These differences, while clinically documented, can be difficult to distinguish visually—especially for automated systems without domain-specific training.

**Solution**: To address this classification problem, I propose a two-pronged approach that evaluates both zero-shot capabilities and fine-tuning potential of state-of-the-art (SOTA) vision-language models. First, I will establish a baseline using Qwen2.5-VL-3B's zero-shot performance to understand how well a powerful multimodal model can classify medical images without task-specific training. Second, I will fine-tune the SigLIP (Sigmoid Loss for Image-Text Pairs) architecture, which employs a Vision Transformer backbone designed for efficient contrastive learning between image and text pairs. The hypothesis is that while foundation models possess general visual understanding, domain-specific fine-tuning remains essential for optimal medical image classification accuracy.
For feature extraction, I will rely on SigLIP's Vision Transformer backbone, which processes image patches through self-attention mechanisms. This generates high-quality image embeddings that capture both fine-grained details (subtle opacities, lesion boundaries) and global context (overall lung structure, bilateral patterns) necessary for distinguishing between visually similar pathologies. The model should learn to focus on diagnostically relevant regions such as lung fields, opacity patterns, and structural abnormalities.
The solution should ideally be agnostic to several factors that vary across medical imaging datasets: scanner manufacturer and model, patient demographics (age, gender), image acquisition parameters, and minor variations in patient positioning. By training on diverse data sources, the classifier should generalize across these variations rather than overfitting to specific imaging conditions.

**Datasets**: I plan to assemble a dataset of approximately 16,300 images across three classes from publicly available medical imaging repositories:

- **COVID CT Slices Dataset**: Provides approximately 14,486 PNG-format images (7,593 COVID-positive and 6,893 normal cases). This dataset offers a large volume of labeled COVID-19 cases for training.
- **MosMedData Chest CT Scans**: Contains 3D volumetric CT scans in NIfTI format with COVID-19 related findings. I will extract center slices from the axial plane, yielding approximately 1,110 2D PNG images. Center slice extraction is chosen because this view typically contains the most comprehensive representation of lung pathology.
- **Chest X-ray Dataset for Tuberculosis**: Provides 704 PNG-format images with tuberculosis positive and negative labels. This dataset addresses the tuberculosis class, though its smaller size presents a class imbalance challenge.

All images will be standardized to 224×224 pixels to meet model input requirements. The data will be split using an 80/10/10 ratio for training, validation, and testing. The training set will be used to optimize model parameters, the validation set will monitor generalization and prevent overfitting, and the test set will remain untouched until final evaluation to provide an unbiased assessment of model performance.
Anticipated Challenges: The significant class imbalance (tuberculosis: ~700 images vs. COVID-19: ~8,400 images) will likely require weighted sampling or other balancing techniques during training. Additionally, the datasets originate from different sources with varying image characteristics, which may introduce domain shift issues that the model must learn to handle.

**Next Steps**: After data acquisition, I will implement preprocessing pipelines to standardize images across sources, evaluate zero-shot baselines, and then proceed with fine-tuning experiments to optimize classification performance.


**Part 2: Data Acquisition and Preparation**
**Overview**
I have assembled a multi-source dataset comprising approximately 16,300 chest imaging samples across three diagnostic classes: COVID-19, tuberculosis, and normal lung cases. The data has been acquired from three publicly available medical imaging repositories, each contributing unique characteristics to the overall dataset.

Dataset Sources are: 
**1. Large COVID-19 CT Scan Slice Dataset**

Source Link: [https://github.com/UCSD-AI4H/COVID-CT](https://www.kaggle.com/datasets/maedemaftouni/large-covid19-ct-slice-dataset)
This one kaggle open-source covid-19 ct dataset contributes 14,486 PNG-format CT slice images, consisting of 7,593 COVID-positive cases and 6,893 normal cases. The original dataset also includes Community-Acquired Pneumonia (CAP) cases (2,618 images), but these were excluded to maintain a clean three-class classification problem focused on COVID-19, tuberculosis, and normal cases.
Sample Characteristics: Images are 2D CT slices in PNG format, capturing axial views of the chest cavity with varying resolutions standardized to 224×224 pixels for model input.

**2. MosMedData Chest CT Scans with COVID-19**

Source Link: [https://mosmed.ai/en/](https://www.kaggle.com/datasets/mathurinache/mosmeddata-chest-ct-scans-with-covid19) 
This is another kaggle open-source dataset originally contains 3D volumetric CT scans in NIfTI format. I extracted center slices along the axial plane from each volumetric scan, yielding 1,110 2D PNG images. The center slice was chosen because it typically provides the most comprehensive view of lung pathology, capturing bilateral lung fields and any central consolidations or opacities.
Preprocessing Applied: For each 3D scan, I applied min-max normalization to ensure consistent contrast and brightness, mapping pixel values to the standard 8-bit range (0-255). The normalized slices were then converted to RGB format for compatibility with pre-trained vision models.

**3. Chest X-ray Dataset for Tuberculosis Segmentation**

Source Link: [https://www.kaggle.com/datasets/raddar/tuberculosis-chest-xrays-shenzhen](https://www.kaggle.com/datasets/iamtapendu/chest-x-ray-lungs-segmentation)
This is the third kaggle open-source dataset, which provides 704 PNG-format chest X-ray images with corresponding labels for tuberculosis positive (PTB=1) and tuberculosis negative (PTB=0) cases. The original dataset includes segmentation masks, but these were excluded from the current study because other datasets lack corresponding masks, and the classification approach does not require explicit region annotations.
Sample Characteristics: Frontal chest X-ray images captured using standard radiographic equipment, standardized to 224×224 pixels.

**Dataset Summary**
Class (Number of Samples Percentage): COVID-19: 8449 - 51.8%; Normal: 7458 - 45.7%; Tuberculosis: 393 - 2.4% with a total of 16,300 samples. 
Data Split
The dataset is partitioned using an 80/10/10 ratio:
Training size: ~13,040 samples; Validation size: ~1,630 samples; and Testsize: ~1,630 samples. 

**Training Set**: Used to optimize model weights through backpropagation. The model sees these samples repeatedly across epochs and learns to extract discriminative features for each class. Due to class imbalance, weighted sampling is applied during training, assigning weights inversely proportional to class frequencies to ensure the model receives adequate exposure to underrepresented classes (particularly tuberculosis).
**Validation Set**: Used to evaluate model performance on unseen data after each training epoch. This subset helps detect overfitting—if training accuracy continues improving while validation accuracy plateaus or decreases, it signals that the model is memorizing training samples rather than learning generalizable patterns. No weighted sampling is applied during validation to obtain realistic performance estimates.
**Key Consideration**: Given the severe class imbalance (tuberculosis represents only 2.4% of total samples), the validation set may contain very few tuberculosis cases (~39 samples). This limited representation makes validation metrics for the tuberculosis class less stable and potentially unreliable as a performance indicator.

**Sample Characteristics**
PropertyDescriptionResolutionAll images standardized to 224×224 pixelsFormatPNG (8-bit RGB)Imaging ModalitiesCT scans (COVID-19, some normal) and X-rays (tuberculosis, some normal)Anatomical ViewAxial CT slices and frontal chest X-raysContrast/BrightnessMin-max normalized to 0-255 range

**Class Imbalance Consideration**
The dataset exhibits significant class imbalance, with tuberculosis severely underrepresented compared to COVID-19 and normal cases. To address this during training, I implemented weighted sampling where each class receives a sampling weight inversely proportional to its frequency. This ensures the model encounters tuberculosis samples more frequently relative to their actual proportion in the dataset, helping prevent the classifier from simply predicting the majority class.


<b></b>Examples in these datasets:

<table id="tfhover" class="tftable" border="1">
<tr><td width="30%"><image src="samples-gif/sample_covid_1.png" /></td><td width="15%"><b>Covid 19 Case</b></td><td>This is one example case that shows the patient has Covid 19 disease.<br /></td></tr>
<tr><td><image src="samples-gif/disgust_07734.gif" /></td><td><b>Disgust</b></td><td>English: A woman looks nervously at her feet. The frown,the closed eyes and  the  open mouth.<br />中文：一个女人紧张的看着脚下的东西。皱眉，眼睛微闭，嘴巴张开。</td></tr>
<tr><td><image src="samples-gif/fear_09246.gif" /></td><td><b>Fear</b></td><td>English: A girl gasps in the dark. The wide eyes and the open mouth.<br />中文：一个女孩在昏暗的环境中急促的喘息。瞪眼，嘴巴张大。</td></tr>
<tr><td><image src="samples-gif/happy_01440.gif" /></td><td><b>Happiness</b></td><td>English: A woman communicates with a man, talking about dinner. The slightly closed eyes, the open mouth and the raised lip corners.<br />中文：一个女人与男人交流，谈论着晚餐。眼睛微闭，嘴巴张开，嘴角上扬。</td></tr>
<tr><td><image src="samples-gif/sad_00467.gif" /></td><td><b>Sadness</b></td><td>English: A girl stands on the beach, tilting her head back and crying. The deep frown and the wide open mouth.<br />中文：一个女孩站在海边，仰着头哭泣。眉头紧蹙，嘴巴张大。</td></tr></table>

<b></b>Categories expect to see within these datasets:

<tr><td width="30%"><image src="samples-gif/example_category.png" /></td><td width="15%"></td></tr>

