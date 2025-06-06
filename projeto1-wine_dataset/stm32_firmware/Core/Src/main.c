/* USER CODE BEGIN Header */
/**
 ******************************************************************************
 * @file           : main.c
 * @brief          : Main program body
 ******************************************************************************
 * @attention
 *
 * Copyright (c) 2025 STMicroelectronics.
 * All rights reserved.
 *
 * This software is licensed under terms that can be found in the LICENSE file
 * in the root directory of this software component.
 * If no LICENSE file comes with this software, it is provided AS-IS.
 *
 ******************************************************************************
 */
/* USER CODE END Header */
/* Includes ------------------------------------------------------------------*/
#include "main.h"
#include "tim.h"
#include "usart.h"
#include "gpio.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */
#include "modelo_convertido.h"
/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */

/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */

/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */

/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/

/* USER CODE BEGIN PV */

/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);
/* USER CODE BEGIN PFP */

/* USER CODE END PFP */

/* Private user code ---------------------------------------------------------*/
/* USER CODE BEGIN 0 */

/* USER CODE END 0 */

/**
 * @brief  The application entry point.
 * @retval int
 */
int main(void)
{

  /* USER CODE BEGIN 1 */

  /* USER CODE END 1 */

  /* MCU Configuration--------------------------------------------------------*/

  /* Reset of all peripherals, Initializes the Flash interface and the Systick. */
  HAL_Init();

  /* USER CODE BEGIN Init */

  /* USER CODE END Init */

  /* Configure the system clock */
  SystemClock_Config();

  /* USER CODE BEGIN SysInit */

  /* USER CODE END SysInit */

  /* Initialize all configured peripherals */
  MX_GPIO_Init();
  MX_USART2_UART_Init();
  MX_TIM1_Init();
  /* USER CODE BEGIN 2 */

  /* USER CODE END 2 */

  /* Infinite loop */
  /* USER CODE BEGIN WHILE */
  float data_input0[30] = {
      -0.55532241f,
      -0.56365376,
      0.20517948,
      0.08933162,
      0.08455566,
      0.11095288,
      1.82918796,
      -0.23589778,
      -0.77348763,
      0.53749837,
      -1.10470039,
      0.19528136,
      0.18376002,
      0.28478338,
      2.44815641,
      0.0875911,
      0.74414693,
      0.22712017,
      -1.39741877,
      -0.5265469,
      0.1474301,
      -0.93655723,
      -0.06999581,
      1.32795285,
      -0.62641772,
      -0.23222238,
      -0.43944822,
      -0.98986549,
      -0.77013498,
      -0.74391404};

  float data_input1[30] = {
      0.44445235f, -0.55789673f, -0.3415184f, -0.3995439f, -0.3795077f, 0.09687179f,
      -1.67683887f, -0.23579761f, 0.65554594f, -1.19838023f, -0.1286992f, -0.06833592f,
      -0.17260604f, -0.05319351f, -1.42408329f, 0.60638318f, 2.57300558f, 0.50863855f,
      0.25125475f, 0.15317924f, 0.60014708f, 2.02392516f, 0.99971423f, -0.41073446f,
      0.3907719f, 0.52176832f, 2.71439431f, -0.74978636f, 0.30822795f, -0.19956318f};

  float data_input2[30] = {
      -0.20246073f, -0.64521166f, -0.37879325f, -0.4257368f, -0.45010014f, -0.28815813f,
      0.43687355f, -0.23612337f, -0.1728245f, -0.37654768f, -0.44480987f, -0.55314457f,
      -0.55144051f, -0.53885779f, 0.06291261f, -0.2762564f, 0.19802942f, -0.18685622f,
      -0.37816145f, -0.5891962f, 0.61109991f, -0.03560273f, 0.15768009f, -0.66742089f,
      -0.42968654f, -0.59297079f, -0.06425726f, -0.31848363f, 0.46165357f, -0.1005903f};

  float data_input3[30] = {
      -0.45137266f, 0.04035445f, -0.62522146f, -0.61418504f, -0.62732588f, -0.41532801f,
      0.23169037f, -0.17260581f, 0.37072268f, 0.80281439f, 0.20958471f, -0.60339662f,
      -0.63335066f, -0.63542262f, -0.21866251f, -0.38565816f, 0.62802892f, -0.39294212f,
      0.04853737f, -0.28172496f, 0.56728859f, -0.19430167f, 0.61036511f, 0.89449181f,
      0.91892805f, 0.9140159f, 0.98199508f, 0.32020065f, 0.35644743f, -0.01227604f};

  char rx_buff[64] = {0};
  char tx_buff[14] = {0};

  // int resultado_teste = modelo_convertido_predict(
  //     data_input0, 30);

  // sprintf(tx_buff, "TESTE0-%d\n", resultado_teste);
  // HAL_UART_Transmit(&huart2, (uint8_t *)tx_buff, 10, 1000);

  // resultado_teste = modelo_convertido_predict(
  //     data_input1, 30);

  // sprintf(tx_buff, "TESTE1-%d\n", resultado_teste);
  // HAL_UART_Transmit(&huart2, (uint8_t *)tx_buff, 10, 1000);

  // resultado_teste = modelo_convertido_predict(
  //     data_input2, 30);

  // sprintf(tx_buff, "TESTE2-%d\n", resultado_teste);
  // HAL_UART_Transmit(&huart2, (uint8_t *)tx_buff, 10, 1000);

  // resultado_teste = modelo_convertido_predict(
  //     data_input3, 30);

  sprintf(tx_buff, "TESTE3-%d\n", sizeof(int));
  HAL_UART_Transmit(&huart2, (uint8_t *)tx_buff, 10, 1000);

  while (1)
  {
    if (HAL_UART_Receive(&huart2, (uint8_t *)rx_buff, 4, 1000) == HAL_OK)
    {
      HAL_GPIO_TogglePin(LD2_GPIO_Port, LD2_Pin);

      uint16_t input[64];
      for (int i = 0; i < 64; i++)
      {
        input[i] = (uint16_t)rx_buff[i];
      }

      int resultado = modelo_convertido_predict(input, 64);
      // sprintf(tx_buff, "ok%d", resultado);
      sprintf(tx_buff, "ok%d", resultado);
      HAL_UART_Transmit(&huart2, (uint8_t *)tx_buff, 3, 1000);
    }
    /* USER CODE END WHILE */

    /* USER CODE BEGIN 3 */
  }
  /* USER CODE END 3 */
}

/**
 * @brief System Clock Configuration
 * @retval None
 */
void SystemClock_Config(void)
{
  RCC_OscInitTypeDef RCC_OscInitStruct = {0};
  RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};

  /** Configure the main internal regulator output voltage
   */
  __HAL_RCC_PWR_CLK_ENABLE();
  __HAL_PWR_VOLTAGESCALING_CONFIG(PWR_REGULATOR_VOLTAGE_SCALE1);

  /** Initializes the RCC Oscillators according to the specified parameters
   * in the RCC_OscInitTypeDef structure.
   */
  RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSI;
  RCC_OscInitStruct.HSIState = RCC_HSI_ON;
  RCC_OscInitStruct.HSICalibrationValue = RCC_HSICALIBRATION_DEFAULT;
  RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
  RCC_OscInitStruct.PLL.PLLSource = RCC_PLLSOURCE_HSI;
  RCC_OscInitStruct.PLL.PLLM = 16;
  RCC_OscInitStruct.PLL.PLLN = 336;
  RCC_OscInitStruct.PLL.PLLP = RCC_PLLP_DIV4;
  RCC_OscInitStruct.PLL.PLLQ = 4;
  if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
  {
    Error_Handler();
  }

  /** Initializes the CPU, AHB and APB buses clocks
   */
  RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK | RCC_CLOCKTYPE_SYSCLK | RCC_CLOCKTYPE_PCLK1 | RCC_CLOCKTYPE_PCLK2;
  RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
  RCC_ClkInitStruct.AHBCLKDivider = RCC_SYSCLK_DIV1;
  RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV2;
  RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV1;

  if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_2) != HAL_OK)
  {
    Error_Handler();
  }
}

/* USER CODE BEGIN 4 */

/* USER CODE END 4 */

/**
 * @brief  This function is executed in case of error occurrence.
 * @retval None
 */
void Error_Handler(void)
{
  /* USER CODE BEGIN Error_Handler_Debug */
  /* User can add his own implementation to report the HAL error return state */
  __disable_irq();
  while (1)
  {
  }
  /* USER CODE END Error_Handler_Debug */
}

#ifdef USE_FULL_ASSERT
/**
 * @brief  Reports the name of the source file and the source line number
 *         where the assert_param error has occurred.
 * @param  file: pointer to the source file name
 * @param  line: assert_param error line source number
 * @retval None
 */
void assert_failed(uint8_t *file, uint32_t line)
{
  /* USER CODE BEGIN 6 */
  /* User can add his own implementation to report the file name and line number,
     ex: printf("Wrong parameters value: file %s on line %d\r\n", file, line) */
  /* USER CODE END 6 */
}
#endif /* USE_FULL_ASSERT */
