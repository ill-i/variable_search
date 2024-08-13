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
