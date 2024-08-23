def correct_vignetting(input_dir, output_dir='output', downscale_factor=50, blur_radius=10):
    # Создаем выходную директорию, если ее не существует
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Проходимся по всем FITS файлам в входной директории
    for filename in os.listdir(input_dir):
        if filename.endswith('.fit') or filename.endswith('.fits'):
            # Открываем FITS файл
            filepath = os.path.join(input_dir, filename)
            with fits.open(filepath) as hdul:
                data = hdul[0].data
                plt.figure()
                plt.imshow(data,cmap="gray")
                # Уменьшаем разрешение кадра
                downsampled_data = data[::downscale_factor, ::downscale_factor]
                plt.figure()
                plt.imshow(downsampled_data,cmap="gray") 
                # Создаем синтетический flat field кадр с использованием гауссового размытия
                synthetic_flat_field = gaussian_filter(downsampled_data, sigma=blur_radius)
                synthetic_flat_field = synthetic_flat_field.astype(np.float64)
                plt.figure()
                plt.imshow(synthetic_flat_field,cmap="gray")
                # Нормируем flat field кадр
                synthetic_flat_field /= np.max(synthetic_flat_field)
                print(np.min(synthetic_flat_field))
                print(np.max(synthetic_flat_field))
                # Интерполируем синтетический flat field обратно к исходному размеру
                synthetic_flat_field_resized = np.repeat(np.repeat(synthetic_flat_field, downscale_factor, axis=0), downscale_factor, axis=1)

                # Убеждаемся, что размеры совпадают
                synthetic_flat_field_resized = synthetic_flat_field_resized[:data.shape[0], :data.shape[1]]

                # Корректируем исходное изображение
                corrected_data = data / synthetic_flat_field_resized
                plt.figure()
                plt.imshow(corrected_data,cmap="gray")
                # Сохраняем синтетический flat field и скорректированное изображение в выходную директорию
                output_flat_field_path = os.path.join(output_dir, f"flat_field_{filename}")
                output_corrected_path = os.path.join(output_dir, f"corrected_{filename}")

                fits.writeto(output_flat_field_path, synthetic_flat_field_resized, hdul[0].header, overwrite=True)
                fits.writeto(output_corrected_path, corrected_data, hdul[0].header, overwrite=True)

    print(f"Процесс завершен. Результаты сохранены в директорию: {output_dir}")
    
    
# mean_data, median_data, std_data = sigma_clipped_stats(data_sub, sigma=3.0)
# data_sub_mask = np.where(data_sub < np.mean(data_sub)+1*std_data,0,1).astype(np.uint8)
# output_filename = filename.split("/")[-1].replace("fit","").replace("fits","")+'_corrected_fits_file.fits'
                
# data_corrected = process_artifacts_and_correct(data_sub, data_sub_mask,output_filename=output_filename, display_results=True)    
    
def process_artifacts_and_correct(data_subtr, data_subtr_mask, output_filename='corrected_fits_file.fits', display_results=True):
    """
    Обрабатывает контуры на изображении, исправляя артефакты на основе площади и циркулярности.
    
    Параметры:
    -----------
    data_subtr : np.ndarray
        Изображение с вычитанным фоном, на котором будут выполняться операции.
    
    data_subtr_mask : np.ndarray
        Маска, по которой будут определяться контуры для анализа.
    
    output_filename : str, optional
        Имя выходного FITS файла с исправленными данными (по умолчанию 'corrected_fits_file.fits').
    
    display_results : bool, optional
        Флаг, указывающий, нужно ли визуализировать исходное и исправленное изображение (по умолчанию True).
    
    Возвращает:
    -----------
    data_corrected : np.ndarray
        Данные изображения после коррекции артефактов.
    """
    # Находим контуры на маске
    contours, _ = cv2.findContours(data_subtr_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Создаем копию данных для замены артефактов
    data_corrected = data_subtr.copy()

    # Вычисляем площади всех контуров и фильтруем их сразу
    areas = [cv2.contourArea(contour) for contour in contours if cv2.contourArea(contour) >= 10]  # 10 — пороговая площадь для исключения маленьких контуров
    if not areas:
        return data_corrected
    
    # Вычисляем медианное значение и стандартное отклонение площади
    mean_area, median_area, std_area = sigma_clipped_stats(areas, sigma=3.0)

    # Отсеиваем контуры, площадь которых меньше медианы плюс 3 стандартных отклонения
    filtered_contours = [contour for contour in contours if cv2.contourArea(contour) >= median_area + 3 * std_area]
    
    # Обработка контуров в параллельных потоках
    def process_contour(contour):
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        circularity = 4 * np.pi * (area / (perimeter ** 2)) if perimeter != 0 else 1
        if circularity < 0.6:  # Порог для линейных объектов, можно варьировать
            cv2.drawContours(data_corrected, [contour], -1, int(np.median(data_subtr)), thickness=3)
    
    with ThreadPoolExecutor() as executor:
        list(tqdm(executor.map(process_contour, filtered_contours), total=len(filtered_contours), desc="Processing contours", unit="contour", miniters=1000, leave=False))

    # Сохраняем исправленный файл (опционально)
    fits.writeto(output_filename, data_corrected, overwrite=True)

    # Визуализация результата
    if display_results:
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        # Гистограмма исходного изображения
        mean_data = np.mean(data_subtr)
        std_data = np.std(data_subtr)
        axs[0, 0].hist(data_subtr.ravel(), bins=256, histtype='step', color='black')
        axs[0, 0].axvline(x=mean_data, color='r', linestyle='--', label=f'Mean: {mean_data:.1f}')
        axs[0, 0].axvline(x=mean_data + std_data, color='g', linestyle='--', label=f'STD: {std_data:.1f}')
        axs[0, 0].axvline(x=mean_data - std_data, color='g', linestyle='--')
        axs[0, 0].set_xlim([mean_data - 5*std_data, mean_data + 5*std_data])
        axs[0, 0].tick_params(axis='x', rotation=45)
        axs[0, 0].set_title('Initial Data Histogram')
        axs[0, 0].set_xlabel('Pixel Value')
        axs[0, 0].set_ylabel('Frequency')
        axs[0, 0].legend(loc="upper right")

        # Гистограмма исправленного изображения
        mean_corrected = np.mean(data_corrected)
        std_corrected = np.std(data_corrected)
        axs[0, 1].hist(data_corrected.ravel(), bins=256, histtype='step', color='black')
        axs[0, 1].axvline(x=mean_corrected, color='r', linestyle='--', label=f'Mean: {mean_corrected:.1f}')
        axs[0, 1].axvline(x=mean_corrected + std_corrected, color='g', linestyle='--', label=f'STD: {std_corrected:.1f}')
        axs[0, 1].axvline(x=mean_corrected - std_corrected, color='g', linestyle='--')
        axs[0, 1].set_xlim([mean_corrected - 5*std_corrected, mean_corrected + 5*std_corrected])
        axs[0, 1].tick_params(axis='x', rotation=45)
        axs[0, 1].set_title('Corrected Data Histogram')
        axs[0, 1].set_xlabel('Pixel Value')
        axs[0, 1].set_ylabel('Frequency')
        axs[0, 1].legend(loc="upper right")

        # Исходное изображение
        axs[1, 0].imshow(data_subtr, cmap='gray', origin='lower')
        axs[1, 0].set_title('Initial Data')
        axs[1, 0].axis('off')

        # Исправленное изображение
        axs[1, 1].imshow(data_corrected, cmap='gray', origin='lower')
        axs[1, 1].set_title('Corrected Data')
        axs[1, 1].axis('off')

        plt.tight_layout()
        plt.show()

    return data_corrected
