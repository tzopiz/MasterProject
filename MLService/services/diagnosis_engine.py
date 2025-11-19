import logging
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)


class DiagnosisEngine:
    """Generates diagnosis and recommendations based on geometric parameters"""
    
    # Normal ranges for TMJ parameters (in mm)
    NORMAL_RANGES = {
        "fossa_height": (8.0, 15.0),
        "head_height": (6.0, 12.0),
        "width": (10.0, 20.0),
        "joint_space": (2.0, 4.0),
        "anterior_space": (1.5, 3.5),
        "posterior_space": (2.0, 4.5),
        "superior_space": (2.0, 4.0)
    }
    
    DISCLAIMER = (
        "Это автоматический анализ, предназначенный только для информационных целей. "
        "Результаты не являются медицинским диагнозом. "
        "Пожалуйста, обратитесь к квалифицированному врачу-стоматологу или челюстно-лицевому хирургу "
        "для профессиональной консультации и точного диагноза."
    )
    
    def diagnose(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate diagnosis based on calculated parameters
        
        Args:
            parameters: Dictionary with geometric parameters
            
        Returns:
            Dictionary with diagnosis information
        """
        logger.info("Generating diagnosis...")
        
        try:
            # Extract parameters
            fossa_height = parameters.get("fossa_height")
            head_height = parameters.get("head_height")
            width = parameters.get("width")
            additional = parameters.get("additional_params", {})
            
            # Analyze parameters
            abnormalities = []
            confidence_scores = []
            
            # Check fossa height
            if fossa_height is not None:
                status, conf = self._check_parameter(
                    "высота суставной ямки", 
                    fossa_height, 
                    self.NORMAL_RANGES["fossa_height"]
                )
                if status != "normal":
                    abnormalities.append(status)
                    confidence_scores.append(conf)
            
            # Check head height
            if head_height is not None:
                status, conf = self._check_parameter(
                    "высота суставной головки",
                    head_height,
                    self.NORMAL_RANGES["head_height"]
                )
                if status != "normal":
                    abnormalities.append(status)
                    confidence_scores.append(conf)
            
            # Check width
            if width is not None:
                status, conf = self._check_parameter(
                    "ширина сустава",
                    width,
                    self.NORMAL_RANGES["width"]
                )
                if status != "normal":
                    abnormalities.append(status)
                    confidence_scores.append(conf)
            
            # Check joint spaces
            for param_name in ["joint_space", "anterior_space", "posterior_space", "superior_space"]:
                if param_name in additional:
                    status, conf = self._check_parameter(
                        param_name.replace("_", " "),
                        additional[param_name],
                        self.NORMAL_RANGES.get(param_name, (0, 100))
                    )
                    if status != "normal":
                        abnormalities.append(status)
                        confidence_scores.append(conf)
            
            # Determine overall status
            if len(abnormalities) == 0:
                overall_status = "normal"
                confidence = 0.85
                recommendations = self._get_normal_recommendations()
            elif len(abnormalities) <= 2:
                overall_status = "minor_abnormality"
                confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.7
                recommendations = self._get_minor_abnormality_recommendations(abnormalities)
            else:
                overall_status = "abnormal"
                confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.75
                recommendations = self._get_abnormal_recommendations(abnormalities)
            
            logger.info(f"Diagnosis: {overall_status}, confidence: {confidence:.2f}")
            
            return {
                "status": overall_status,
                "confidence": confidence,
                "recommendations": recommendations,
                "disclaimer": self.DISCLAIMER,
                "abnormalities": abnormalities if abnormalities else None
            }
            
        except Exception as e:
            logger.error(f"Error generating diagnosis: {str(e)}", exc_info=True)
            return {
                "status": "error",
                "confidence": 0.0,
                "recommendations": ["Ошибка при анализе данных"],
                "disclaimer": self.DISCLAIMER
            }
    
    def _check_parameter(
        self, 
        param_name: str, 
        value: float, 
        normal_range: tuple
    ) -> tuple[str, float]:
        """
        Check if parameter is within normal range
        
        Args:
            param_name: Name of parameter
            value: Measured value
            normal_range: Tuple of (min, max) normal values
            
        Returns:
            Tuple of (status_message, confidence)
        """
        min_val, max_val = normal_range
        
        if min_val <= value <= max_val:
            return ("normal", 0.9)
        elif value < min_val:
            deviation = (min_val - value) / min_val
            confidence = min(0.9, 0.5 + deviation * 0.4)
            return (f"{param_name} ниже нормы ({value:.1f}mm)", confidence)
        else:  # value > max_val
            deviation = (value - max_val) / max_val
            confidence = min(0.9, 0.5 + deviation * 0.4)
            return (f"{param_name} выше нормы ({value:.1f}mm)", confidence)
    
    def _get_normal_recommendations(self) -> List[str]:
        """Get recommendations for normal TMJ"""
        return [
            "Параметры височно-нижнечелюстного сустава находятся в пределах нормы",
            "Рекомендуется поддерживать хорошую гигиену полости рта",
            "Избегайте чрезмерной нагрузки на челюсть (жевание твердой пищи, жевательная резинка)",
            "При появлении дискомфорта или боли обратитесь к стоматологу",
            "Регулярные профилактические осмотры (раз в 6-12 месяцев)"
        ]
    
    def _get_minor_abnormality_recommendations(self, abnormalities: List[str]) -> List[str]:
        """Get recommendations for minor abnormalities"""
        recommendations = [
            "Обнаружены незначительные отклонения от нормы:",
        ]
        recommendations.extend([f"  - {abn}" for abn in abnormalities])
        recommendations.extend([
            "",
            "Рекомендации:",
            "• Обратитесь к стоматологу для детальной консультации",
            "• Может потребоваться дополнительная диагностика",
            "• Избегайте факторов, увеличивающих нагрузку на ВНЧС",
            "• Рассмотрите возможность использования каппы при бруксизме",
            "• Физиотерапия и упражнения для челюсти могут быть полезны"
        ])
        return recommendations
    
    def _get_abnormal_recommendations(self, abnormalities: List[str]) -> List[str]:
        """Get recommendations for significant abnormalities"""
        recommendations = [
            "⚠️ Обнаружены значительные отклонения от нормы:",
        ]
        recommendations.extend([f"  - {abn}" for abn in abnormalities])
        recommendations.extend([
            "",
            "Настоятельно рекомендуется:",
            "• СРОЧНО обратитесь к челюстно-лицевому хирургу или стоматологу",
            "• Необходима профессиональная консультация и дополнительная диагностика",
            "• Может потребоваться МРТ для оценки мягких тканей",
            "• Не откладывайте визит к врачу - раннее вмешательство важно",
            "• Избегайте нагрузки на челюсть до консультации с врачом",
            "",
            "Возможные методы лечения (определяет врач):",
            "• Консервативное лечение (физиотерапия, медикаменты)",
            "• Ортопедическое лечение (каппы, шины)",
            "• Хирургическое вмешательство (в сложных случаях)"
        ])
        return recommendations
    
    def calculate_asymmetry(
        self,
        left_params: Dict[str, float],
        right_params: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Calculate asymmetry between left and right TMJ
        
        Args:
            left_params: Parameters for left TMJ
            right_params: Parameters for right TMJ
            
        Returns:
            Dictionary with asymmetry analysis
        """
        try:
            asymmetries = {}
            
            for param in ["fossa_height", "head_height", "width"]:
                if param in left_params and param in right_params:
                    left_val = left_params[param]
                    right_val = right_params[param]
                    
                    # Calculate percentage difference
                    avg = (left_val + right_val) / 2
                    diff_percent = abs(left_val - right_val) / avg * 100
                    
                    asymmetries[param] = {
                        "left": left_val,
                        "right": right_val,
                        "difference_percent": diff_percent,
                        "significant": diff_percent > 10  # >10% is significant
                    }
            
            # Overall asymmetry assessment
            significant_asymmetries = sum(
                1 for a in asymmetries.values() if a.get("significant", False)
            )
            
            if significant_asymmetries == 0:
                status = "symmetric"
            elif significant_asymmetries <= 1:
                status = "mild_asymmetry"
            else:
                status = "significant_asymmetry"
            
            return {
                "status": status,
                "asymmetries": asymmetries,
                "significant_count": significant_asymmetries
            }
            
        except Exception as e:
            logger.error(f"Error calculating asymmetry: {str(e)}")
            return {"status": "error", "asymmetries": {}}

