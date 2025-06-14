// /* USER CODE BEGIN Header */
// /**
//  ******************************************************************************
//  * @file           : main.c
//  * @brief          : Main program body
//  ******************************************************************************
//  * @attention
//  *
//  * Copyright (c) 2025 STMicroelectronics.
//  * All rights reserved.
//  *
//  * This software is licensed under terms that can be found in the LICENSE file
//  * in the root directory of this software component.
//  * If no LICENSE file comes with this software, it is provided AS-IS.
//  *
//  ******************************************************************************
//  */
// /* USER CODE END Header */
// /* Includes ------------------------------------------------------------------*/
// #include "main.h"
// #include "tim.h"
// #include "usart.h"
// #include "gpio.h"
// #include "core_cm4.h" // For Cortex-M4 core

// /* Private includes ----------------------------------------------------------*/
// /* USER CODE BEGIN Includes */
// #include "modelo_convertido.h"
// #include <string.h>
// /* USER CODE END Includes */

// /* Private typedef -----------------------------------------------------------*/
// /* USER CODE BEGIN PTD */

// /* USER CODE END PTD */

// /* Private define ------------------------------------------------------------*/
// /* USER CODE BEGIN PD */
// #define PARAMS 31
// #define PARAMS_SIZE (PARAMS * sizeof(float))

// /* USER CODE END PD */

// /* Private macro -------------------------------------------------------------*/
// /* USER CODE BEGIN PM */

// /* USER CODE END PM */

// /* Private variables ---------------------------------------------------------*/

// /* USER CODE BEGIN PV */

// /* USER CODE END PV */

// /* Private function prototypes -----------------------------------------------*/
// void SystemClock_Config(void);
// /* USER CODE BEGIN PFP */

// /* USER CODE END PFP */

// /* Private user code ---------------------------------------------------------*/
// /* USER CODE BEGIN 0 */

// /* USER CODE END 0 */

// /**
//  * @brief  The application entry point.
//  * @retval int
//  */
// int main(void)
// {

//   /* USER CODE BEGIN 1 */

//   /* USER CODE END 1 */

//   /* MCU Configuration--------------------------------------------------------*/

//   /* Reset of all peripherals, Initializes the Flash interface and the Systick. */
//   HAL_Init();

//   /* USER CODE BEGIN Init */

//   /* USER CODE END Init */

//   /* Configure the system clock */
//   SystemClock_Config();

//   /* USER CODE BEGIN SysInit */

//   /* USER CODE END SysInit */

//   /* Initialize all configured peripherals */
//   MX_GPIO_Init();
//   MX_USART2_UART_Init();
//   MX_TIM1_Init();
//   /* USER CODE BEGIN 2 */

//   // Enable the cycle counter
//   CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
//   DWT->CYCCNT = 0;
//   DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;

//   /* USER CODE END 2 */

//   /* Infinite loop */
//   /* USER CODE BEGIN WHILE */

//   char rx_buff[PARAMS_SIZE] = {0};
//   float decoded_floats[PARAMS]; // Array to store the decoded floats
//   char tx_buff[20] = {0};

//   while (1)
//   {
//     if (HAL_UART_Receive(&huart2, (uint8_t *)rx_buff, PARAMS_SIZE, 1000) == HAL_OK)
//     {
//       HAL_GPIO_TogglePin(LD2_GPIO_Port, LD2_Pin);

//       // Converte a string recebida em floats
//       for (int i = 0; i < PARAMS; i++)
//       {
//         memcpy(&decoded_floats[i], &rx_buff[i * sizeof(float)], sizeof(float));
//       }

//       volatile uint32_t start_cycles = DWT->CYCCNT;

//       // Execute prediction
//       int resultado = modelo_convertido_predict(decoded_floats, PARAMS);

//       // Get elapsed cycles
//       volatile uint32_t elapsed_cycles = DWT->CYCCNT - start_cycles;

//       sprintf(tx_buff, "ok%d:%06lu", resultado, elapsed_cycles);
//       HAL_UART_Transmit(&huart2, (uint8_t *)tx_buff, strlen(tx_buff), 1000);
//     }
//     /* USER CODE END WHILE */

//     /* USER CODE BEGIN 3 */
//   }
//   /* USER CODE END 3 */
// }

// /**
//  * @brief System Clock Configuration
//  * @retval None
//  */
// void SystemClock_Config(void)
// {
//   RCC_OscInitTypeDef RCC_OscInitStruct = {0};
//   RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};

//   /** Configure the main internal regulator output voltage
//    */
//   __HAL_RCC_PWR_CLK_ENABLE();
//   __HAL_PWR_VOLTAGESCALING_CONFIG(PWR_REGULATOR_VOLTAGE_SCALE1);

//   /** Initializes the RCC Oscillators according to the specified parameters
//    * in the RCC_OscInitTypeDef structure.
//    */
//   RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSI;
//   RCC_OscInitStruct.HSIState = RCC_HSI_ON;
//   RCC_OscInitStruct.HSICalibrationValue = RCC_HSICALIBRATION_DEFAULT;
//   RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
//   RCC_OscInitStruct.PLL.PLLSource = RCC_PLLSOURCE_HSI;
//   RCC_OscInitStruct.PLL.PLLM = 16;
//   RCC_OscInitStruct.PLL.PLLN = 336;
//   RCC_OscInitStruct.PLL.PLLP = RCC_PLLP_DIV4;
//   RCC_OscInitStruct.PLL.PLLQ = 4;
//   if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
//   {
//     Error_Handler();
//   }

//   /** Initializes the CPU, AHB and APB buses clocks
//    */
//   RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK | RCC_CLOCKTYPE_SYSCLK | RCC_CLOCKTYPE_PCLK1 | RCC_CLOCKTYPE_PCLK2;
//   RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
//   RCC_ClkInitStruct.AHBCLKDivider = RCC_SYSCLK_DIV1;
//   RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV2;
//   RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV1;

//   if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_2) != HAL_OK)
//   {
//     Error_Handler();
//   }
// }

// /* USER CODE BEGIN 4 */

// /* USER CODE END 4 */

// /**
//  * @brief  This function is executed in case of error occurrence.
//  * @retval None
//  */
// void Error_Handler(void)
// {
//   /* USER CODE BEGIN Error_Handler_Debug */
//   /* User can add his own implementation to report the HAL error return state */
//   __disable_irq();
//   while (1)
//   {
//   }
//   /* USER CODE END Error_Handler_Debug */
// }

// #ifdef USE_FULL_ASSERT
// /**
//  * @brief  Reports the name of the source file and the source line number
//  *         where the assert_param error has occurred.
//  * @param  file: pointer to the source file name
//  * @param  line: assert_param error line source number
//  * @retval None
//  */
// void assert_failed(uint8_t *file, uint32_t line)
// {
//   /* USER CODE BEGIN 6 */
//   /* User can add his own implementation to report the file name and line number,
//      ex: printf("Wrong parameters value: file %s on line %d\r\n", file, line) */
//   /* USER CODE END 6 */
// }
// #endif /* USE_FULL_ASSERT */
