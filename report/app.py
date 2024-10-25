import datetime
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

@dataclass
class PatientInfo:
    name: Optional[str] = None
    age: Optional[int] = None
    gender: Optional[str] = None
    referring_physician: Optional[str] = None
    medical_history: Optional[str] = None
    symptoms: Optional[List[str]] = None

class XRayReportGenerator:
    def __init__(self, model_path: str, confidence_threshold: float = 0.75):
        """Initialize the report generator with a trained TensorFlow model."""
        self.model = tf.keras.models.load_model(model_path)
        self.model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        self.confidence_threshold = confidence_threshold
        self.report_patterns = self._initialize_report_patterns()

    def _initialize_report_patterns(self) -> Dict:
        """Initialize report templates and patterns."""
        return {
            "Normal": {
                "findings": {
                    "lung_fields": "Lung fields appear clear without definitive consolidation, infiltrates, or effusions.",
                    "lung_volumes": "Lung volumes appear adequate with visible costophrenic angles.",
                    "cardiovascular": "Heart size and mediastinal contours appear within normal limits.",
                    "pleural_space": "No definitive evidence of pleural effusion or pneumothorax.",
                    "bones": "No acute osseous abnormalities identified.",
                    "soft_tissues": "Soft tissues appear unremarkable."
                },
                "impression": "No definitive acute cardiopulmonary findings identified on this examination.",
                "recommendations": [
                    "Clinical correlation is recommended",
                    "Consider follow-up imaging if symptoms persist or worsen",
                    "Compare with prior studies if available"
                ]
            },
            "Pneumonia": {
                "findings": {
                    "lung_opacity": {
                        "severe": "Areas of increased opacity noted in the lung fields, potentially representing severe consolidation patterns.",
                        "moderate": "Patchy areas of increased opacity noted, possibly representing moderate consolidation.",
                        "mild": "Subtle areas of increased opacity noted, may represent early or mild consolidation."
                    },
                    "cardiovascular": {
                        "normal": "Cardiac silhouette appears within normal limits. Mediastinal contours are preserved.",
                        "abnormal": "Cardiac silhouette appears mildly enlarged. Further evaluation of mediastinal structures may be warranted."
                    },
                    "pleural_space": {
                        "severe": "Possible significant pleural effusion noted with blunting of costophrenic angles.",
                        "moderate": "Possible moderate pleural effusion with blunting of costophrenic angles.",
                        "mild": "Minimal blunting of costophrenic angles noted, may represent small pleural effusion.",
                        "normal": "Costophrenic angles appear preserved."
                    }
                },
                "severity_recommendations": {
                    "severe": [
                        "Clinical correlation strongly recommended",
                        "Consider additional imaging studies for confirmation",
                        "Monitor clinical status closely",
                        "Consider infectious disease consultation if confirmed",
                        "Follow-up imaging recommended based on clinical course"
                    ],
                    "moderate": [
                        "Clinical correlation recommended",
                        "Consider follow-up imaging in 24-48 hours if symptoms persist",
                        "Monitor for clinical improvement",
                        "Consider additional diagnostic testing if clinically indicated",
                        "Compare with prior studies if available"
                    ],
                    "mild": [
                        "Clinical correlation recommended",
                        "Consider follow-up imaging if symptoms worsen",
                        "Monitor clinical course",
                        "Compare with prior studies if available",
                        "Consider additional views if clinically indicated"
                    ]
                }
            },
            "Uncertain": {
                "findings": "The radiographic findings are indeterminate. Technical factors or patient positioning may limit interpretation.",
                "impression": "Findings are inconclusive and require clinical correlation and possibly additional imaging.",
                "recommendations": [
                    "Clinical correlation is essential",
                    "Consider additional views or imaging modalities",
                    "Compare with prior studies if available",
                    "Follow-up imaging may be warranted based on clinical presentation"
                ]
            }
        }

    def preprocess_image(self, image_path: str) -> Tuple[np.ndarray, bool]:
        """Preprocess the X-ray image for model input with quality checks."""
        try:
            target_size = (224, 224)
            img = Image.open(image_path).convert('RGB')
            
            # Image quality checks
            if img.size[0] < 200 or img.size[1] < 200:
                return None, False
                
            # Check for proper positioning and exposure
            img_array = np.array(img)
            mean_intensity = np.mean(img_array)
            std_intensity = np.std(img_array)
            
            # Check if image is too dark or too bright
            if mean_intensity < 30 or mean_intensity > 225:
                return None, False
            
            # Check if image has enough contrast
            if std_intensity < 10:
                return None, False
                
            img = img.resize(target_size)
            img_array = np.array(img)
            img_array = img_array / 255.0
            return np.expand_dims(img_array, axis=0), True
            
        except Exception as e:
            print(f"Error preprocessing image: {str(e)}")
            return None, False

    def _analyze_affected_areas(self, img: np.ndarray) -> Dict[str, float]:
        """Analyze which areas of the lung are affected."""
        h, w = img.shape
        
        # Divide image into regions
        left_upper = img[:h//2, :w//2]
        left_lower = img[h//2:, :w//2]
        right_upper = img[:h//2, w//2:]
        right_lower = img[h//2:, w//2:]
        
        # Calculate mean intensity for each region
        regions = {
            "Left Upper Lobe": np.mean(left_upper),
            "Left Lower Lobe": np.mean(left_lower),
            "Right Upper Lobe": np.mean(right_upper),
            "Right Lower Lobe": np.mean(right_lower)
        }
        
        # Calculate opacity percentages
        baseline = np.percentile(img, 95)
        opacity_threshold = baseline * 0.7
        
        opacity_percentages = {}
        for region_name, region_intensity in regions.items():
            opacity_percentages[region_name] = 100 * (1 - (region_intensity / baseline))
        
        # Find most affected area
        most_affected = min(regions.items(), key=lambda x: x[1])[0]
        
        return {
            "primary_affected_area": most_affected,
            "opacity_percentages": opacity_percentages,
            "opacity_percentage": np.mean(list(opacity_percentages.values()))
        }

    def _analyze_pleural_space(self, img: np.ndarray) -> Dict[str, str]:
        """Analyze pleural space condition."""
        h, w = img.shape
        
        # Analyze lower regions of both lungs
        left_lower = img[int(h*0.7):, :w//2]
        right_lower = img[int(h*0.7):, w//2:]
        
        # Calculate mean intensities
        left_intensity = np.mean(left_lower)
        right_intensity = np.mean(right_lower)
        
        def get_condition(intensity):
            if intensity < 100:
                return "severe"
            elif intensity < 150:
                return "moderate"
            elif intensity < 200:
                return "mild"
            else:
                return "normal"
                
        left_condition = get_condition(left_intensity)
        right_condition = get_condition(right_intensity)
        
        severity_order = ["normal", "mild", "moderate", "severe"]
        overall_condition = max(left_condition, right_condition, 
                              key=lambda x: severity_order.index(x))
        
        return {
            "overall_condition": overall_condition,
            "left_pleural_space": left_condition,
            "right_pleural_space": right_condition
        }

    def _determine_severity(self, confidence_score: float, image_features: Optional[Dict] = None) -> str:
        """Determine disease severity based on confidence score and image features."""
        # Base severity on confidence score
        base_severity = "mild"
        if confidence_score > 0.9:
            base_severity = "severe"
        elif confidence_score > 0.8:
            base_severity = "moderate"
            
        if image_features is None:
            return base_severity
            
        # Adjust severity based on additional features
        severity_score = 0
        
        if 'opacity_percentage' in image_features:
            opacity = image_features['opacity_percentage']
            if opacity > 30:
                severity_score += 2
            elif opacity > 15:
                severity_score += 1
                
        if 'pleural_condition' in image_features:
            if image_features['pleural_condition'] == 'severe':
                severity_score += 2
            elif image_features['pleural_condition'] == 'moderate':
                severity_score += 1
                
        if severity_score >= 3:
            return "severe"
        elif severity_score >= 1:
            return "moderate"
        else:
            return base_severity

    def analyze_image(self, image_path: str) -> Tuple[str, float, Dict]:
        """Analyze the X-ray image using the loaded model."""
        # Preprocess image with validation
        processed_img, is_valid = self.preprocess_image(image_path)
        if not is_valid:
            return "Uncertain", 0.0, {"error": "Invalid or poor quality image"}
        
        # Get model prediction with multiple samples
        predictions = []
        for _ in range(5):
            pred = self.model.predict(processed_img, verbose=0)[0][0]
            predictions.append(pred)
        
        mean_pred = np.mean(predictions)
        std_pred = np.std(predictions)
        
        if std_pred > 0.15:
            return "Uncertain", float(mean_pred), {"uncertainty": "High prediction variance"}
        
        if mean_pred > self.confidence_threshold:
            classification = "Pneumonia"
            confidence = mean_pred
        elif mean_pred < (1 - self.confidence_threshold):
            classification = "Normal"
            confidence = 1 - mean_pred
        else:
            return "Uncertain", float(mean_pred), {"uncertainty": "Low confidence"}
        
        # Additional analysis for positive cases
        if classification == "Pneumonia":
            affected_areas = self._analyze_affected_areas(cv2.imread(image_path, 0))
            pleural_analysis = self._analyze_pleural_space(cv2.imread(image_path, 0))
            
            severity = self._determine_severity(confidence, {
                'opacity_percentage': affected_areas['opacity_percentage'],
                'pleural_condition': pleural_analysis['overall_condition']
            })
            
            analysis = {
                "severity": severity,
                "affected_areas": affected_areas,
                "pleural_condition": pleural_analysis['overall_condition'],
                "confidence_metrics": {
                    "mean_prediction": float(mean_pred),
                    "std_prediction": float(std_pred)
                }
            }
        else:
            analysis = {
                "condition": "normal",
                "confidence_metrics": {
                    "mean_prediction": float(mean_pred),
                    "std_prediction": float(std_pred)
                }
            }
        
        return classification, float(confidence), analysis

    def generate_report(self, image_path: str, patient_info: PatientInfo = PatientInfo()) -> str:
        """
        Generate a medical report based on image analysis with uncertainty handling.
        """
        # Analyze image
        classification, confidence, analysis = self.analyze_image(image_path)
        
        report = []
        
        # Header
        report.append("RADIOLOGICAL REPORT")
        report.append("=" * 50)
        report.append(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}")
        
        # Patient Information
        report.append("\nPATIENT INFORMATION")
        report.append("-" * 20)
        report.append(f"Name: {patient_info.name or 'Not Provided'}")
        report.append(f"Age: {patient_info.age or 'Not Provided'}")
        report.append(f"Gender: {patient_info.gender or 'Not Provided'}")
        report.append(f"Referring Physician: {patient_info.referring_physician or 'Not Provided'}")
        
        if patient_info.medical_history:
            report.append(f"Relevant Medical History: {patient_info.medical_history}")
        if patient_info.symptoms:
            report.append(f"Presenting Symptoms: {', '.join(patient_info.symptoms)}")
        
        # Study Information
        report.append("\nSTUDY INFORMATION")
        report.append("-" * 20)
        report.append("Study: Chest X-ray (PA View)")
        report.append(f"Analysis: {classification}")
        
        if classification != "Uncertain":
            report.append(f"AI Analysis Confidence: {confidence*100:.1f}%")
            if "confidence_metrics" in analysis:
                report.append(f"Prediction Stability: ±{analysis['confidence_metrics']['std_prediction']*100:.1f}%")
        
        # Findings and Recommendations
        report.append("\nFINDINGS")
        report.append("-" * 20)
        
        if classification == "Uncertain":
            report.append(self.report_patterns["Uncertain"]["findings"])
            report.append("\nIMPRESSION")
            report.append("-" * 20)
            report.append(self.report_patterns["Uncertain"]["impression"])
            report.append("\nRECOMMENDATIONS")
            report.append("-" * 20)
            for i, rec in enumerate(self.report_patterns["Uncertain"]["recommendations"], 1):
                report.append(f"{i}. {rec}")
        elif classification == "Normal":
            for finding in self.report_patterns["Normal"]["findings"].values():
                report.append(finding)
            report.append("\nIMPRESSION")
            report.append("-" * 20)
            report.append(self.report_patterns["Normal"]["impression"])
            report.append("\nRECOMMENDATIONS")
            report.append("-" * 20)
            for i, rec in enumerate(self.report_patterns["Normal"]["recommendations"], 1):
                report.append(f"{i}. {rec}")
        else:  # Pneumonia case
            pattern = self.report_patterns["Pneumonia"]
            report.append(f"Lung Opacity: {pattern['findings']['lung_opacity'][analysis['severity']]}")
            report.append(f"Location: Predominantly affecting the {analysis['affected_areas']}")
            report.append(f"Pleural Space: {pattern['findings']['pleural_space'][analysis['pleural_condition']]}")
            report.append(pattern['findings']['cardiovascular']['normal'])
            
            report.append("\nIMPRESSION")
            report.append("-" * 20)
            report.append(f"Findings suggestive of {analysis['severity'].title()} Pneumonia affecting the {analysis['affected_areas']}")
            report.append("Clinical correlation is strongly recommended.")
            
            report.append("\nRECOMMENDATIONS")
            report.append("-" * 20)
            for i, rec in enumerate(pattern['severity_recommendations'][analysis['severity']], 1):
                report.append(f"{i}. {rec}")
        
        # Add disclaimer
        report.append("\nDISCLAIMER")
        report.append("-" * 20)
        report.append("This report was generated with the assistance of AI technology and should be reviewed by a qualified healthcare professional. Clinical correlation is essential.")
        
        return "\n".join(report)
# Example usage
if __name__ == "__main__":
    # Initialize generator with model path
    generator = XRayReportGenerator("D:/IMP/Agile_Avengers/train/pneumonia/pneumonia_model.h5",confidence_threshold=0.75)

    patient_info = PatientInfo(
        medical_history="No prior respiratory issues",
        symptoms=["cough", "fever"]
        )

    
    # Generate report
    report = generator.generate_report("D:/IMP/Agile_Avengers/train/archive(1)/chest_xray/train/NORMAL/IM-0156-0001.jpeg")
    print(report)