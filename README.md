# PWT-API_ConnectionUsb (UNIDOS E RS232 Interface)

## Descripción
Esta aplicación permite la comunicación con el dispositivo UNIDOS E a través de una interfaz RS232, facilitando la adquisición y el guardado de mediciones de dosis y tasa de dosis. La aplicación está desarrollada en Python utilizando PySide6 para la interfaz gráfica y `pyserial` para la comunicación con el puerto serie.

## Requisitos
Antes de ejecutar la aplicación, asegúrate de tener instaladas las siguientes dependencias:

```bash
pip install PySide6 pyserial
```

## Uso

1. Conecta el dispositivo UNIDOS E a un puerto serie de tu computadora.
2. Ejecuta la aplicación:
   
   ```bash
   python PTW.py
   ```

3. En la interfaz gráfica:
   - Ingresa el puerto serie correspondiente (por defecto `COM7`).
   - Presiona "Connect" para establecer la comunicación con el dispositivo.
   - Selecciona el modo de medición (`Dose` o `Dose Rate`).
   - Presiona "Start Measurement" para comenzar la adquisición de datos.
   - Presiona "Stop Measurement" para detener la adquisición.
   - Usa el botón "Save Data" para guardar los datos adquiridos en un archivo de texto.
   - Puedes cambiar el rango de medición utilizando el botón "Change Range".

## Formato de Datos Guardados
Los datos se guardan en un archivo de texto con el siguiente formato:

```
Time, Dose, Unit
00:01, 0.123, Gy
00:02, 0.125, Gy
...
```

## Contribuciones
Las contribuciones son bienvenidas. Si encuentras algún problema o deseas mejorar la aplicación, crea un issue o un pull request en este repositorio.

## Licencia
Este proyecto está bajo la Apache License.

