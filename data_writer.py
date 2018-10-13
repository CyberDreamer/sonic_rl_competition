# -*- coding: utf-8 -*-
import cv2


class VideoWriter(object):
    def __init__(self, width, height, fps, path):
        """Инициализирует буфер для кадров и создает объект
        cv2.video_writer для записи видео потока
        в файл посредством указанных кодеков"""
        self.frames = []
        self.width = width
        self.height = height
        codec = cv2.VideoWriter_fourcc('M', 'P', '4', '2')

        self.video_writer = cv2.VideoWriter(path, codec, fps, (width, height))

    def add_frame(self, frame):
        """Добавляет новый кадр в очередь (буфер) на запись в файл.
        При накоплении в буфере N кадров происходит их запись
        на диск и очистка буфера"""
        frame = cv2.resize(frame.copy(), dsize=(self.width, self.height))
        self.frames.append(frame)

        if len(self.frames) >= 300:
            self.frame_flush()

    def frame_flush(self):
        for frame in self.frames:
            self.video_writer.write(frame)
        self.frames = []

    def stop_and_release(self):
        """Дописывает имеющиеся и не сохраненные кадры в файл.
        После чего очищает память кадров и вызывает деструктор
        объекта cv2.video_writer"""
        if self.frames:
            self.frame_flush()

        self.video_writer.release()
